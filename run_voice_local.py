"""
Standalone Local Live Voice Bot Runner for AeroAssist.
Uses Laptop Microphone for Audio In and Laptop Speakers for Audio Out.
Powered by:
  - STT: SpeechRecognition (Google Free Speech-to-Text with Hindi & English support)
  - LLM: Google Gemini 2.5 Flash (via google-genai SDK) with Flight Tools
  - TTS: Edge-TTS (hi-IN-SwaraNeural) + Pygame Mixer
"""

import os
import sys
import time
import uuid
import asyncio
import tempfile
import logging
from pathlib import Path

# Force UTF-8 on Windows console for emoji and Hindi text support
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

# Suppress Pygame welcome banner
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import edge_tts
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import numpy as np
import io
from google import genai
from google.genai import types

# Import application business logic and prompts
from services.flight_service import FlightService
from agent.state_manager import ConversationState
from agent.escalation_handler import EscalationHandler
from config.prompts import AEROASSIST_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Setup & Configuration
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
TTS_VOICE = "hi-IN-SwaraNeural"  # Natural Indian voice supporting Hindi & Hinglish

# Initialize Flight Service & State
flight_service = FlightService()
state = ConversationState()

# Initialize Pygame Mixer for Speaker Playback
try:
    pygame.mixer.init()
except Exception as e:
    print(f"⚠️ Pygame mixer warning: {e}")


# ---------------------------------------------------------------------------
# TTS: Text-To-Speech (Audio Out)
# ---------------------------------------------------------------------------
def speak(text: str) -> None:
    """
    Synthesize text into speech via Edge TTS and play through laptop speakers.
    """
    clean_text = text.strip()
    if not clean_text:
        return

    print(f"\n🔊 AeroAssist: {clean_text}\n")

    async def _synth(output_path: str):
        communicate = edge_tts.Communicate(clean_text, voice=TTS_VOICE)
        await communicate.save(output_path)

    # Use unique temp file to avoid Windows file locks
    temp_file = os.path.join(tempfile.gettempdir(), f"aeroassist_{uuid.uuid4().hex}.mp3")
    try:
        asyncio.run(_synth(temp_file))
        if os.path.exists(temp_file):
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            clock = pygame.time.Clock()
            while pygame.mixer.music.get_busy():
                clock.tick(15)
            pygame.mixer.music.unload()
    except Exception as e:
        print(f"⚠️ [TTS Playback Error]: {e}")
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass


import re

def normalize_airline_speech(text: str) -> str:
    """
    Normalize Hinglish / Hindi speech to ensure terms like PNR and flight codes
    are properly captured in English alphanumeric format (e.g. converting 'नर' or 'पीएनआर' to 'PNR').
    """
    if not text:
        return text

    # Convert Devanagari numerals to English digits
    hindi_to_eng_digits = {
        "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
        "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"
    }
    for h_dig, e_dig in hindi_to_eng_digits.items():
        text = text.replace(h_dig, e_dig)

    # Phonetic mappings for PNR in Devanagari / Hinglish
    pnr_variations = [
        "पी एन आर", "पीएनआर", "पी.एन.आर.", "पी एन आर नंबर",
        "p n r", "p.n.r", "pee n r", "pea n r", "p and r", "pianar"
    ]
    for var in pnr_variations:
        text = re.sub(re.escape(var), "PNR", text, flags=re.IGNORECASE)

    # When spoken quickly in Hindi, Google STT often misinterprets "PNR" as "नर"
    text = re.sub(r'\bनर\s*(नंबर|code|hai|batao|\d|[a-zA-Z])', r'PNR \1', text, flags=re.IGNORECASE)
    text = re.sub(r'(मेरा|apna|check|karo|batao)\s*नर\b', r'\1 PNR', text, flags=re.IGNORECASE)

    # Flight airline code fixes in Devanagari
    text = re.sub(r'\bसिक्स\s*ई\b', '6E', text, flags=re.IGNORECASE)
    text = re.sub(r'\b6\s*ई\b', '6E', text, flags=re.IGNORECASE)
    text = re.sub(r'\bए\s*आई\b', 'AI', text, flags=re.IGNORECASE)
    text = re.sub(r'\bक्यु\s*पी\b', 'QP', text, flags=re.IGNORECASE)

    return text.strip()


def transcribe_with_gemini(client, audio_bytes: bytes) -> str:
    """
    Use Gemini 3.6 Flash multimodal audio understanding to transcribe speech
    with high accuracy when standard STT fails.
    """
    if not client or not audio_bytes:
        return ""
    try:
        part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
        prompt = (
            "You are a speech-to-text transcriber for flight customer support. "
            "Transcribe the spoken words in this audio exactly in Hindi/English/Hinglish. "
            "Accurately capture airline codes and PNR numbers. "
            "Output ONLY the transcribed words. If there is no human speech or only noise, output empty."
        )
        resp = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[prompt, part],
        )
        if resp.text:
            text = resp.text.strip().strip('"').strip("'")
            if text.lower() in ["empty", "none", "silence", "no speech", "nothing"]:
                return ""
            return text
        return ""
    except Exception as e:
        return ""


def record_audio_with_autoboost(sample_rate=16000, max_duration=10.0, silence_timeout=1.6):
    """
    Record user speech using sounddevice with live audio level meter,
    automatic silence detection, and software gain auto-boost (up to 50x).
    """
    chunk_duration = 0.2  # 200ms per slice
    chunk_samples = int(chunk_duration * sample_rate)
    max_chunks = int(max_duration / chunk_duration)
    max_silence_chunks = int(silence_timeout / chunk_duration)

    print("🎙️ Listening... (Boliye, microphone sun raha hai)")

    chunks = []
    has_speech_started = False
    silence_chunks = 0
    speech_threshold = 0.003  # Sensitive threshold for laptop mic

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
            for i in range(max_chunks):
                chunk, _ = stream.read(chunk_samples)
                chunk_peak = float(np.max(np.abs(chunk)))
                chunks.append(chunk)

                # Real-time visual audio meter
                bar_len = min(int(chunk_peak * 250), 20)
                meter = "█" * bar_len + "░" * (20 - bar_len)
                print(f"\r  Mic Level: [{meter}]", end="", flush=True)

                if chunk_peak > speech_threshold:
                    has_speech_started = True
                    silence_chunks = 0
                elif has_speech_started:
                    silence_chunks += 1
                    if silence_chunks >= max_silence_chunks:
                        break
                else:
                    # If silence for 5 seconds before speaking, finish
                    if i >= int(5.0 / chunk_duration):
                        break
    except Exception as e:
        print(f"\n⚠️ Microphone capture error: {e}")
        return None

    print("\n⏳ Processing audio...")
    if not chunks:
        return None

    full_audio = np.concatenate(chunks, axis=0)
    overall_peak = float(np.max(np.abs(full_audio)))

    # If pure background noise / complete silence
    if overall_peak < 0.001:
        print("⏳ Silence detected (koi aawaaz nahi aayi).")
        return None

    # Apply Dynamic Software Gain Auto-Boost
    gain = min(0.85 / overall_peak, 50.0)
    boosted_audio = full_audio * gain
    if gain > 1.5:
        print(f"🔊 Audio auto-boosted {gain:.1f}x for crystal-clear recognition.")

    bio = io.BytesIO()
    sf.write(bio, boosted_audio, sample_rate, format='WAV', subtype='PCM_16')
    return bio.getvalue()


# ---------------------------------------------------------------------------
# STT: Speech-To-Text (Audio In)
# ---------------------------------------------------------------------------
def listen_to_user(recognizer: sr.Recognizer, gemini_client=None) -> str:
    """
    Listen to user speech with automatic gain boosting and transcribe via:
    1. Google STT (hi-IN) with PNR normalization
    2. Google STT (en-IN) with PNR normalization
    3. Gemini 3.6 Flash Multimodal Audio Decoding
    """
    try:
        wav_bytes = record_audio_with_autoboost()
    except Exception as e:
        print(f"⚠️ Mic error: {e}")
        wav_bytes = None

    if not wav_bytes:
        try:
            print("💡 Tip: Agar mic se aawaaz nahi aa rahi, toh yahan type bhi kar sakte hain:")
            raw = input("💬 You (Type or press Enter to retry mic): ").strip()
            return normalize_airline_speech(raw)
        except (EOFError, KeyboardInterrupt):
            return "exit"

    # 1. Transcribe using SpeechRecognition on amplified WAV
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio = recognizer.record(source)

        # 1a. Hindi / Hinglish
        try:
            text = recognizer.recognize_google(audio, language="hi-IN")
            text = normalize_airline_speech(text)
            print(f"🎙️ You said: {text}")
            return text
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"⚠️ [Google STT Network Warning]: {e}")

        # 1b. English (India)
        try:
            text = recognizer.recognize_google(audio, language="en-IN")
            text = normalize_airline_speech(text)
            print(f"🎙️ You said: {text}")
            return text
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            pass

    except Exception as e:
        print(f"⚠️ STT pre-processing error: {e}")

    # 2. Fallback to Gemini 3.6 Flash Native Audio Model
    if gemini_client:
        try:
            print("🤖 Asking Gemini 3.6 Flash to decode speech...")
            text = transcribe_with_gemini(gemini_client, wav_bytes)
            if text:
                text = normalize_airline_speech(text)
                print(f"🎙️ You said (via Gemini): {text}")
                return text
        except Exception:
            pass

    print("❓ Audio samajh nahi aaya. Kripya thoda tez aur saaf bolein.")
    return ""


# ---------------------------------------------------------------------------
# Tool Definitions for Gemini Function Calling
# ---------------------------------------------------------------------------
def lookup_booking(pnr: str) -> str:
    """Look up flight booking details using the customer's 6-character PNR code.

    Args:
        pnr: The 6-character flight booking reference (e.g., 6E2849, AI805X, QP1102, 6E9921).
    """
    booking = flight_service.get_booking(pnr)
    if not booking:
        state.record_tool("lookup_booking", f"PNR {pnr} not found")
        return f"Maaf kijiye, PNR {pnr} humare system mein nahi mila. Kripya ek baar dobara check karein."

    state.verified_pnr = booking.pnr
    state.passenger_name = booking.passenger_name
    state.record_tool("lookup_booking", f"Found {booking.pnr} for {booking.passenger_name}")

    status_str = booking.flight.status.value.lower().replace("_", " ")
    return (
        f"Aapki booking mil gayi hai. Passenger {booking.passenger_name}, "
        f"flight {booking.flight.flight_number} from {booking.flight.origin} to {booking.flight.destination}, "
        f"flight status {status_str} hai."
    )


def check_flight_status(flight_number: str) -> str:
    """Check the real-time operational status of an airline flight.

    Args:
        flight_number: Airline flight number (e.g., 6E-2049, AI-805, QP-1102, 6E-8834).
    """
    segment = flight_service.check_flight_status(flight_number)
    if not segment:
        state.record_tool("check_flight_status", f"Flight {flight_number} not found")
        return f"Maaf kijiye, flight {flight_number} ki jaankari nahi mili. Kripya flight number check karein."

    dep_time_str = segment.departure_time.strftime("%I:%M %p")
    status_val = segment.status.value.lower().replace("_", " ")
    state.record_tool("check_flight_status", f"{segment.flight_number}: {status_val}")
    return f"Flight {segment.flight_number} {status_val} hai, scheduled departure time {dep_time_str} hai."


def calculate_reschedule_quote(pnr: str, new_flight_number: str, new_date: str = "") -> str:
    """Calculate the cost and fees for rescheduling an existing booking to a new flight.

    Args:
        pnr: The customer's booking PNR.
        new_flight_number: Target flight number to reschedule to (e.g., 6E-2051).
        new_date: Optional desired date for the new flight.
    """
    quote = flight_service.calculate_reschedule_quote(pnr, new_flight_number)
    if not quote:
        state.record_tool("calculate_reschedule_quote", f"Failed for PNR {pnr}")
        return "Maaf kijiye, reschedule quote calculate nahi ho paya. Kripya PNR aur flight number check karein."

    diff = int(quote.fare_difference)
    fee = int(quote.reschedule_fee)
    total = int(quote.total_payable)
    state.record_tool("calculate_reschedule_quote", f"Quote total ₹{total} for PNR {pnr}")
    return (
        f"Reschedule ke liye fare difference {diff} rupees aur reschedule fee {fee} rupees, "
        f"total {total} rupees payable hoga."
    )


def calculate_refund(pnr: str) -> str:
    """Calculate the refund estimate and cancellation penalty for cancelling a flight booking.

    Args:
        pnr: The customer's booking PNR to estimate refund for.
    """
    refund = flight_service.calculate_refund(pnr)
    if not refund:
        state.record_tool("calculate_refund", f"Failed for PNR {pnr}")
        return "Maaf kijiye, refund calculate nahi ho saka. Kripya PNR dobara check karein."

    penalty = int(refund.cancellation_penalty)
    refundable = int(refund.refundable_amount)
    state.record_tool("calculate_refund", f"Refund ₹{refundable}, penalty ₹{penalty}")
    if penalty == 0:
        return f"Flight airline dwara cancel hone ke kaaran aapko poora 100 percent refund yaani {refundable} rupees milega."
    return f"Standard cancellation penalty {penalty} rupees ke baad aapko {refundable} rupees refund milega."


def escalate_to_human(reason: str) -> str:
    """Trigger a warm transfer escalation to a senior customer support executive.

    Args:
        reason: The reason for escalating to a human supervisor.
    """
    state.escalation_triggered = True
    state.escalation_reason = reason
    state.record_tool("escalate_to_human", f"Reason: {reason}")
    return "Main aapki call turant humare human flight executive ko transfer kar rahi hoon. Kripya hold karein."


FLIGHT_TOOLS = [
    lookup_booking,
    check_flight_status,
    calculate_reschedule_quote,
    calculate_refund,
    escalate_to_human,
]


# ---------------------------------------------------------------------------
# Main Voice Loop
# ---------------------------------------------------------------------------
def main():
    banner = """
==================================================
✈️  AeroAssist LIVE VOICE BOT (Mic In + Speaker Out)
Powered by Gemini (gemini-3.6-flash) + Flight Database
==================================================
"""
    print(banner)

    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY is not set.")
        print("Please add your key in .env file:")
        print("   GEMINI_API_KEY=your_actual_gemini_key\n")
        sys.exit(1)

    # 1. Initialize Gemini Client with Tools and System Instruction
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(
        system_instruction=AEROASSIST_SYSTEM_PROMPT,
        tools=FLIGHT_TOOLS,
        temperature=0.7,
    )

    model_candidates = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    chat = None
    selected_model = None

    for candidate in model_candidates:
        try:
            chat = client.chats.create(
                model=candidate,
                config=config,
            )
            selected_model = candidate
            print(f"🤖 Connected to model: {selected_model}\n")
            break
        except Exception as e:
            print(f"⚠️ Could not load {candidate}: {e}. Trying fallback...")

    if chat is None:
        print("❌ Error: Failed to connect to any Gemini model candidate.")
        sys.exit(1)

    # 2. Initialize Speech Engine
    recognizer = sr.Recognizer()
    print("✅ Microphone engine ready with real-time visual meter & 50x gain auto-boost!\n")

    # 3. Agent Speaks First Greeting
    initial_greeting = (
        "Namaste! AeroAssist flight customer support mein aapka swagat hai. "
        "Main aapki kya madad kar sakti hoon? Kripya apna PNR number bataiye."
    )
    speak(initial_greeting)

    # 4. Interactive Voice Loop
    while True:
        try:
            # Capture speech from user (with Auto-Gain Boost + Google STT + Gemini STT)
            user_text = listen_to_user(recognizer, gemini_client=client)

            if not user_text:
                continue

            # Check for exit commands
            if user_text.lower() in ["exit", "quit", "bye", "alvida", "stop"]:
                farewell = "Dhanyavaad! AeroAssist ko call karne ke liye shukriya. Have a great flight!"
                speak(farewell)
                break

            # Update conversation turn & frustration scoring
            state.update_turn(user_text)

            # Check if escalation criteria met prior to sending to LLM
            if state.should_escalate():
                hold_msg = EscalationHandler.generate_hold_message()
                speak(hold_msg)
                print("\n" + state.get_summary_for_handoff() + "\n")
                break

            # Send speech transcript to Gemini LLM
            print("🤖 AeroAssist is thinking...")
            response = chat.send_message(user_text)

            # Check if an escalation tool was triggered by the model
            if state.should_escalate():
                hold_msg = EscalationHandler.generate_hold_message()
                speak(hold_msg)
                print("\n" + state.get_summary_for_handoff() + "\n")
                break

            # Speak Gemini's final synthesized response
            if response.text:
                speak(response.text)

        except KeyboardInterrupt:
            print("\n\n👋 Voice Bot stopped by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ [Unexpected Error]: {e}\n")
            time.sleep(1)


if __name__ == "__main__":
    main()

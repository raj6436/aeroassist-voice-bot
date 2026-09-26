"""
Microphone & Audio Diagnostic Tool for AeroAssist.
Tests if your microphone is capturing audio, displays volume levels,
plays back what was recorded, and tests speech transcription.
"""

import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os

# Set UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 60)
print("🎙️  AeroAssist Microphone & Audio Diagnostic Tool")
print("=" * 60)

# 1. List available input devices
devices = sd.query_devices()
default_in, default_out = sd.default.device
print(f"\nDefault Input Device : [{default_in}] {devices[default_in]['name']}")
print(f"Default Output Device: [{default_out}] {devices[default_out]['name']}\n")

print("Available Microphones:")
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        marker = "-> [ACTIVE DEFAULT]" if i == default_in else ""
        print(f"  [{i}] {d['name']} {marker}")

print("\n" + "-" * 60)
print("STEP 1: Recording 4 seconds of audio from your microphone...")
print("👉 PLEASE SPEAK LOUDLY INTO YOUR MIC: 'Mera PNR 6E2849 hai'")
print("-" * 60)

for count in range(3, 0, -1):
    print(f"Starting in {count}...", end="\r")
    time.sleep(1)

print("\n🔴 RECORDING NOW... (Speak loudly!)")

sample_rate = 16000
duration = 4.0
try:
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    # Show progress dots
    for _ in range(4):
        time.sleep(1)
        print("  ...", end="", flush=True)
    sd.wait()
    print("\n✅ Recording finished!\n")
except Exception as e:
    print(f"\n❌ Recording failed: {e}")
    sys.exit(1)

# Analyze audio volume
rms = float(np.sqrt(np.mean(recording**2)))
peak = float(np.max(np.abs(recording)))
print("=" * 60)
print("AUDIO ANALYSIS RESULTS:")
print(f"  Peak Volume Level: {peak:.5f}")
print(f"  RMS Volume Level : {rms:.5f}")

if peak < 0.01:
    print("\n❌ PROBLEM DETECTED: MICROPHONE IS RECEIVING PURE SILENCE!")
    print("------------------------------------------------------------")
    print("Possible reasons:")
    print("1. Laptop microphone hardware mute button is ON (check F4 / Fn+F4).")
    print("2. Windows Privacy Settings is blocking microphone access:")
    print("   -> Go to: Windows Settings > Privacy & security > Microphone")
    print("   -> Turn ON: 'Let apps access your microphone'")
    print("   -> Turn ON: 'Let desktop apps access your microphone'")
    print("3. Windows Microphone Volume is muted or set to 0%:")
    print("   -> Go to: Windows Settings > System > Sound > Microphone")
    print("   -> Check Input Volume slider (set to 80-100%).")
    print("4. Windows is using a disconnected Bluetooth headset as default.")
    print("------------------------------------------------------------")
else:
    print("\n✅ MICROPHONE IS WORKING! Audio signal detected.")

# Playback recorded audio through speakers
print("\n" + "-" * 60)
print("STEP 2: Playing back your recorded audio through speakers...")
print("-" * 60)
try:
    sd.play(recording, sample_rate)
    sd.wait()
    print("✅ Playback complete. Did you hear your own voice clearly?")
except Exception as e:
    print(f"⚠️ Playback warning: {e}")

# Save to temp WAV and test Google STT
temp_wav = os.path.join(tempfile.gettempdir(), "test_mic_output.wav")
sf.write(temp_wav, recording, sample_rate)

print("\n" + "-" * 60)
print("STEP 3: Testing Google Speech Recognition on this recording...")
print("-" * 60)

try:
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.AudioFile(temp_wav) as source:
        audio_data = r.record(source)

    try:
        text_hi = r.recognize_google(audio_data, language="hi-IN")
        print(f"  Transcribed (hi-IN): '{text_hi}'")
    except sr.UnknownValueError:
        print("  hi-IN: Could not understand audio")
    except Exception as e:
        print(f"  hi-IN Error: {e}")

    try:
        text_en = r.recognize_google(audio_data, language="en-IN")
        print(f"  Transcribed (en-IN): '{text_en}'")
    except sr.UnknownValueError:
        print("  en-IN: Could not understand audio")
    except Exception as e:
        print(f"  en-IN Error: {e}")

except Exception as e:
    print(f"⚠️ STT test warning: {e}")
finally:
    if os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
        except Exception:
            pass

print("\n" + "=" * 60)
print("Diagnostic test completed.")
print("=" * 60)

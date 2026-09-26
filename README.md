# AeroAssist - AI Flight Support Voice Agent

A real-time AI voice agent for airline customer support featuring conversational English and Hinglish support, direct business tool execution (bookings, flight status, reschedule quotes, refunds), real-time sentiment/frustration tracking, and warm human escalation.

Built with **LiveKit Agents**, **Deepgram STT**, **Google Gemini LLM**, **Cartesia TTS**, and **Silero VAD**.

---

## Architecture

```
                                  +------------------------------------+
                                  |      Customer (Web / Phone / SIP)  |
                                  +------------------------------------+
                                                    |
                                          WebRTC Audio Stream
                                                    v
                                  +------------------------------------+
                                  |         LiveKit Room / Cloud       |
                                  +------------------------------------+
                                                    |
                                    +---------------+---------------+
                                    |                               |
                                    v                               v
                         +--------------------+           +--------------------+
                         |  Silero VAD        |           |  Deepgram STT      |
                         |  (Voice Activity   |           |  (Hindi / Hinglish |
                         |   Detection)       |           |   Transcription)   |
                         +--------------------+           +--------------------+
                                    |                               |
                                    +---------------+---------------+
                                                    v
                                  +------------------------------------+
                                  |      AeroAssist Voice Pipeline     |
                                  |      (agent/core_agent.py)         |
                                  +------------------------------------+
                                         |                      ^
                         State & Context |                      | Hinglish Speech
                                         v                      | Synthesis
                         +--------------------+           +--------------------+
                         | ConversationState  |           | Cartesia TTS       |
                         | (Turn & Frustration|           | (Low-Latency Voice |
                         |  Scoring)          |           |  Generation)       |
                         +--------------------+           +--------------------+
                                         |
                                         v
                         +------------------------------------+
                         |    Google Gemini 2.5 Flash LLM     |
                         |    (Function Calling & Reasoning)  |
                         +------------------------------------+
                                         |
                   +---------------------+---------------------+
                   v                                           v
       +-----------------------+                   +-----------------------+
       | Flight Service Tools  |                   |  Escalation Handler   |
       | - lookup_booking      |                   |  - Frustration Score  |
       | - check_flight_status |                   |  - Hold Announcement  |
       | - reschedule_quote    |                   |  - Warm Transfer Card |
       | - calculate_refund    |                   +-----------------------+
       +-----------------------+                               |
                   |                                           v
                   v                               +-----------------------+
       +-----------------------+                   | Senior Support Agent  |
       | In-Memory Mock DB /   |                   | (Human Handoff)       |
       | Airline Core API      |                   +-----------------------+
       +-----------------------+
```

---

## Tech Stack

- **Framework**: [LiveKit Agents Python SDK](https://github.com/livekit/agents)
- **VAD (Voice Activity Detection)**: [Silero VAD](https://github.com/snakers4/silero-vad)
- **STT (Speech-to-Text)**: [Deepgram](https://deepgram.com/) (`model="nova-3"`, `language="hi"`)
- **LLM**: [Google Gemini 2.5 Flash](https://ai.google.dev/) via `livekit-plugins-google`
- **TTS (Text-to-Speech)**: [Cartesia](https://cartesia.ai/) Sonic low-latency voice synthesis
- **Backend & Data**: Python 3.13, Pydantic v2, FastAPI, Uvicorn

---

## Project Structure

```
Flight_Support_VoiceBot/
├── agent/
│   ├── __init__.py               # Package init & version compatibility shims
│   ├── core_agent.py             # Main LiveKit VoicePipelineAgent entrypoint
│   ├── escalation_handler.py     # Evaluation & warm transfer briefing generator
│   ├── state_manager.py          # ConversationState & frustration scoring
│   └── tools.py                  # LLM Function Tools (lookup, status, quotes, refund)
├── config/
│   ├── __init__.py
│   ├── prompts.py                # AeroAssist voice persona & Hinglish prompt
│   └── settings.py               # Pydantic Settings loaded from .env
├── database/
│   ├── __init__.py
│   ├── mock_db.py                # Realistic domestic bookings (IndiGo, Air India, Akasa)
│   └── schema.py                 # Pydantic models for segments, bookings, refunds
├── services/
│   ├── __init__.py
│   ├── flight_service.py         # Business logic for PNR, status, quotes, refunds
│   └── summary_service.py        # Structured warm-transfer briefing formatter
├── tests/
│   ├── __init__.py
│   ├── test_flight_services.py   # Unit tests for database and service layers
│   └── test_agent_pipeline.py    # Unit tests for state, escalation, and tools
├── .env.example                  # Environment configuration template
├── requirements.txt              # Project dependencies
├── run_local.py                  # Development CLI runner
└── README.md
```

---

## Setup & Installation

### 1. Create and Activate Virtual Environment

```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

```ini
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
DEEPGRAM_API_KEY=your_deepgram_api_key
GEMINI_API_KEY=your_google_gemini_api_key
CARTESIA_API_KEY=your_cartesia_api_key
```

---

## Running the Agent

### Start in Development Mode

```bash
python run_local.py dev
```

Connect using the [LiveKit Agents Playground](https://agents-playground.livekit.io/) or the LiveKit CLI to start talking to AeroAssist.

---

## Testing

Run the full automated test suite (27 unit tests):

```bash
python -m unittest discover -s tests -v
```

All tests pass covering:
- PNR lookup (valid, invalid, case-insensitive)
- Flight status checking (on-time, delayed, cancelled)
- Reschedule quotes (fare diff + ₹500 fee)
- Refund rules (₹2,500 penalty vs 100% refund for airline cancellations)
- ConversationState tracking & frustration scoring
- Escalation thresholds & hold announcements
- Function tool execution & state updates


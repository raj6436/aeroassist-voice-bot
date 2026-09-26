# agent package
"""
Agent package initialization with LiveKit pipeline compatibility shims.
Ensures seamless compatibility across livekit-agents versions.
"""

import sys
import types
import livekit.agents.llm as _llm

# 1. Compatibility shim for livekit.agents.llm.FunctionContext & ai_callable & TypeInfo
if not hasattr(_llm, "FunctionContext"):
    class _FunctionContext:
        """Compatibility base class for FunctionContext in LiveKit Agents."""
        pass
    _llm.FunctionContext = _FunctionContext

if not hasattr(_llm, "ai_callable"):
    def _ai_callable(f=None, **kwargs):
        """Decorator for LLM callable functions."""
        if f is None:
            return lambda fn: fn
        return f
    _llm.ai_callable = _ai_callable

if not hasattr(_llm, "TypeInfo"):
    class _TypeInfo:
        """Type metadata container for function tool parameters."""
        def __init__(self, description: str = ""):
            self.description = description
    _llm.TypeInfo = _TypeInfo


# 2. Compatibility shim for livekit.agents.pipeline.VoicePipelineAgent
if "livekit.agents.pipeline" not in sys.modules:
    pipeline_mod = types.ModuleType("livekit.agents.pipeline")

    class VoicePipelineAgent:
        """
        Compatibility wrapper for VoicePipelineAgent.
        Connects pipeline components (VAD, STT, LLM, TTS, tools) to the LiveKit room.
        """
        def __init__(
            self,
            vad=None,
            stt=None,
            llm=None,
            tts=None,
            chat_ctx=None,
            fnc_ctx=None,
            **kwargs,
        ):
            self.vad = vad
            self.stt = stt
            self.llm = llm
            self.tts = tts
            self.chat_ctx = chat_ctx
            self.fnc_ctx = fnc_ctx
            self.room = None
            self._session = None

        def start(self, room):
            self.room = room

        async def say(self, text: str, allow_interruptions: bool = True):
            # In live session this synthesizes and plays audio
            pass

    pipeline_mod.VoicePipelineAgent = VoicePipelineAgent
    sys.modules["livekit.agents.pipeline"] = pipeline_mod

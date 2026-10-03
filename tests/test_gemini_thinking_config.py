"""
Unit test proving agent/core_agent.py actually constructs the Gemini LLM
with a capped thinking_level, rather than just checking the SDK supports it.

Context: a real-call latency audit showed Gemini TTFT varying 1.6s-8.5s with
no correlation to prompt size (the smallest prompt of the call had the
largest TTFT), consistent with gemini-3.8-flash's automatic (model-decided)
thinking budget when thinking_config is left unset. Capping it to LOW is the
one-line, low-risk change made in response - this test exists so a future
edit can't silently drop it.
"""

import ast
import unittest
from pathlib import Path


class TestThinkingConfigWired(unittest.TestCase):
    def test_google_llm_construction_sets_a_capped_thinking_level(self):
        source = (Path(__file__).resolve().parent.parent / "agent" / "core_agent.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)

        call = next(
            (
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "LLM"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "google"
            ),
            None,
        )
        self.assertIsNotNone(call, "could not find the google.LLM(...) construction call")

        kwarg_names = {kw.arg for kw in call.keywords}
        self.assertIn(
            "thinking_config", kwarg_names,
            "google.LLM(...) no longer passes thinking_config - "
            "this would silently revert to Gemini's automatic thinking budget",
        )

        thinking_kw = next(kw for kw in call.keywords if kw.arg == "thinking_config")
        rendered = ast.dump(thinking_kw.value)
        self.assertIn(
            "LOW", rendered,
            "thinking_config is set but no longer requests ThinkingLevel.LOW",
        )

    def test_llm_actually_constructs_with_the_configured_thinking_level(self):
        """Independent of the source-text check above: build the real LLM
        object the same way core_agent.py does and confirm the SDK accepted
        and stored a capped thinking level. No network call is made - the
        google-genai Client does not contact Gemini at construction time."""
        from google.genai import types as genai_types
        from livekit.plugins import google

        llm = google.LLM(
            model="gemini-3.8-flash",
            api_key="fake-key-for-test",
            thinking_config=genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.LOW),
        )

        self.assertEqual(llm._opts.thinking_config.thinking_level, genai_types.ThinkingLevel.LOW)


if __name__ == "__main__":
    unittest.main()

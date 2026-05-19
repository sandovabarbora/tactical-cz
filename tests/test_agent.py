"""Tests for the Phase 4 LLM agent.

The Anthropic client is mocked — these tests verify context-building
correctness (the project-specific logic), not Claude's behaviour.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _tiny_events_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "events.parquet"
    rows = []
    for f, p_shot in [(50, 0.2), (100, 0.85), (150, 0.95), (200, 0.3)]:
        rows.append({"frame_idx": f, "second": f / 25.0,
                     "event_type": "Shot", "confidence": p_shot,
                     "source_video": "synth.mp4"})
        rows.append({"frame_idx": f, "second": f / 25.0,
                     "event_type": "Pass", "confidence": 0.5,
                     "source_video": "synth.mp4"})
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_events_summary_lists_per_class_max() -> None:
    from tactical_cz.agent.qa import _events_summary
    events = pd.DataFrame([
        {"frame_idx": 50, "second": 2.0, "event_type": "Shot", "confidence": 0.95},
        {"frame_idx": 60, "second": 2.4, "event_type": "Shot", "confidence": 0.10},
        {"frame_idx": 50, "second": 2.0, "event_type": "Pass", "confidence": 0.30},
    ])
    summary = _events_summary(events)
    assert "Shot: max=0.950" in summary
    assert "at t=2.0s" in summary
    assert "Pass: max=0.300" in summary


def test_build_context_handles_events_only(tmp_path: Path) -> None:
    """build_context should work even when vision + transcript are missing."""
    from tactical_cz.agent.qa import build_context
    events_path = _tiny_events_parquet(tmp_path)
    ctx = build_context(events_path, vision_path=None, transcript_path=None)
    assert "Shot" in ctx.events_summary
    assert "unavailable" in ctx.tactical_summary
    assert "unavailable" in ctx.commentary_summary
    assert ctx.clip_duration_s > 0


def test_clip_context_as_prompt_includes_all_sections(tmp_path: Path) -> None:
    from tactical_cz.agent.qa import build_context
    events_path = _tiny_events_parquet(tmp_path)
    prompt = build_context(events_path).as_prompt()
    assert "EVENTS MODEL" in prompt
    assert "TACTICAL FEATURES" in prompt
    assert "CZECH COMMENTARY" in prompt


def test_clip_qa_raises_without_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Also disable .env auto-load by pointing dotenv at empty file
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    events_path = _tiny_events_parquet(tmp_path)
    from tactical_cz.agent.qa import ClipQA
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not set"):
        ClipQA(events_path=events_path)


def test_clip_qa_ask_returns_anthropic_response(tmp_path: Path, monkeypatch) -> None:
    """Mock Anthropic client; verify ask() returns concatenated text content."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")
    events_path = _tiny_events_parquet(tmp_path)

    # Fake response shape: client.messages.create returns an object whose
    # .content is a list of blocks with .text attributes.
    fake_block = MagicMock()
    fake_block.text = "Model fires Shot strongest at t=6.0s with P=0.95."
    fake_resp = MagicMock()
    fake_resp.content = [fake_block]

    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = fake_resp

        from tactical_cz.agent.qa import ClipQA
        qa = ClipQA(events_path=events_path)
        answer = qa.ask("When does Shot peak?")

        assert "0.95" in answer
        # Verify Claude was called with our system prompt + context
        call = instance.messages.create.call_args
        assert "BAS classes" in call.kwargs["system"]
        assert "Shot" in call.kwargs["system"]

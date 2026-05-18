"""tactical-cz — multimodal AI pipeline for Czech football broadcasts.

Five layers: vision (detection/tracking) → events (action spotting)
→ audio (Czech ASR) → multimodal alignment → LLM agent (Czech tactical QA).

Public surface is via the demo (Gradio / HF Space); modules here are
the building blocks. Module-level functions, no class hierarchy.
"""

__version__ = "0.1.0"

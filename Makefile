# tactical-cz — phase-driven Makefile
#
# Each phase has its own target; cross-phase data flows through
# data/processed/ (gitignored — too big). Heavy training targets are
# annotated with the expected GPU + cost. Run those on RunPod / Vast.

PY := uv run python
PYTEST := uv run pytest

.PHONY: help install install-dev install-train \
        vision events audio align agent demo all \
        notebook test lint format clean

help:
	@echo "tactical-cz — make targets"
	@echo ""
	@echo "  install            Sync core dependencies"
	@echo "  install-dev        Sync with dev extras (pytest, ruff, jupyter)"
	@echo "  install-train      Sync with training extras (lightning, wandb)"
	@echo ""
	@echo "  vision             Phase 1: run Roboflow-sports vision baseline"
	@echo "                       on data/raw/match.mp4 → data/processed/vision/"
	@echo "  events             Phase 2: run MatchVision/SoccerMaster fine-tune"
	@echo "                       GPU: A100 80GB, ~12hrs (~\$10 RunPod community)"
	@echo "  audio              Phase 3: Whisper CZ transcription + diarization"
	@echo "  align              Phase 4: join events ↔ transcript ↔ tactical patterns"
	@echo "  agent              Phase 5: LLM agent over enriched event DB"
	@echo "  demo               Phase 6: launch local Gradio demo"
	@echo "  all                run phases 1→6 sequentially (assumes data + models)"
	@echo ""
	@echo "  notebook           start Jupyter for ad-hoc exploration"
	@echo "  test               run offline tests"
	@echo "  lint               ruff check"
	@echo "  format             ruff format"
	@echo "  clean              remove data/processed/* and outputs/*"

install:
	uv sync

install-dev:
	uv sync --extra dev

install-train:
	uv sync --extra dev --extra train

# ─── Phase entry points (stubs — wired in subsequent phases) ──────────────

vision:
	@if [ -z "$(INPUT)" ]; then \
	  echo "Usage: make vision INPUT=path/to/match.mp4 [MAX_FRAMES=300]"; \
	  exit 1; \
	fi
	$(PY) -m tactical_cz.vision --input $(INPUT) \
	  $(if $(MAX_FRAMES),--max-frames $(MAX_FRAMES),) \
	  $(if $(STRIDE),--stride $(STRIDE),)

events:
	@echo "Phase 2 not yet implemented"
	@echo "Will run: $(PY) -m tactical_cz.events --features data/processed/vision/"

audio:
	@echo "Phase 3 not yet implemented"
	@echo "Will run: $(PY) -m tactical_cz.audio --input data/raw/match.mp4"

align:
	@echo "Phase 4 not yet implemented"
	@echo "Will run: $(PY) -m tactical_cz.multimodal --events ... --transcript ..."

agent:
	@echo "Phase 5 not yet implemented"
	@echo "Will run: $(PY) -m tactical_cz.agent --enriched data/processed/aligned/"

demo:
	@echo "Phase 6 not yet implemented"
	@echo "Will run: $(PY) -m tactical_cz.demo"

all: vision events audio align agent demo

# ─── Dev helpers ──────────────────────────────────────────────────────────

notebook:
	uv run jupyter lab --notebook-dir notebooks

test:
	$(PYTEST) -m "not network and not gpu"

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	rm -rf data/processed/* data/interim/* outputs/*

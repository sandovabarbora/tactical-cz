# Phase 2 Kickoff — Events Layer

**Goal:** action spotting on broadcast video (goal, shot, pass, corner, foul, card, throw-in). Fine-tune MatchVision / SoccerMaster on SoccerNet's Ball Action Spotting (BAS) task; emit per-clip event timeline parquet.

**Timeline:** Phase 2 = weeks 3-5 of the 14-week plan in [PRD.md § Implementation phases](../PRD.md#implementation-phases-14-weeks-part-time).

This doc lists what to set up before fine-tuning. **The 2026-05-18 revision removed the SoccerNet NDA step** that the original draft assumed: the BAS 2025 task is now hosted as a public HuggingFace dataset, no paperwork.

---

## 1. SoccerNet BAS 2025 — two-stage gate

Half-NDA: the data is **fetchable** without paperwork (HF public dataset, GPL-3.0), but the zips are **AES-encrypted** with the historical SoccerNet NDA password. Concretely:

| Stage | Gated? |
|---|---|
| HF download (`SoccerNet/SN-BAS-2025`) | **public, no paperwork** |
| AES extraction of zip → mp4 + Labels-ball.json | **NDA password required** |

Get the password by filling the SoccerNet NDA Google Form (~5 minutes, no academic affiliation required, response usually within 24-72h):
https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform

Once approved:

```bash
export SOCCERNET_PASSWORD=<password from approval email>
# (Or pass --password explicitly; env var keeps it out of shell history + git diff)
uv run python -m tactical_cz.events.downloader extract --split valid
```

Total ~19 GB across train/valid/test/challenge — fits on most laptop SSDs, no need to download directly to GPU instance.

### What we actually need

Only the **Ball Action Spotting (BAS) 2025 task**: clips with frame-level event labels (goal, shot, pass, corner, free-kick, throw-in, foul, card, drive, save, etc.). Skip everything else (action spotting full corpus, GSR, MVFouls — separate downloads, would balloon to 2 TB).

### Sizes

| Split | Size |
|---|---|
| train.zip | 8.5 GB |
| valid.zip | 2.0 GB |
| test.zip | 4.5 GB |
| challenge.zip | 4.2 GB |
| **Total** | **~19 GB** |

Prototype against `valid` first (smaller, fast iteration). Pull `train` only when ready to fine-tune.

### Steps

```bash
# Recommended path: use our wrapper (logs + cache layout)
uv run python -m tactical_cz.events.downloader --splits valid          # 2 GB warm-up
uv run python -m tactical_cz.events.downloader --splits train valid    # full fine-tune set

# Or call SoccerNet directly if you want
uv run python -c "
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory='data/raw/soccernet')
d.downloadDataTask(task='spotting-ball-2025', split=['valid'])
"
```

### License (GPL-3.0)

SoccerNet ships BAS 2025 under GPL-3.0. Our code stays MIT (no derivative-of-data condition triggers for source), but **fine-tuned weights we publish are derived from GPL-3.0 training data** and inherit GPL-3.0 distribution obligations. Plan:

- Code in this repo: **MIT** (unchanged)
- Fine-tuned model checkpoints we publish to HuggingFace Hub: **GPL-3.0**, dual-licensed if we ever want commercial use
- Add a short licence note in the project README and on the HF model card before publishing the first checkpoint

---

## 2. RunPod (or Vast.ai) account + first A100 instance

Local Mac MPS works for inference (Phase 1 vision baseline). For Phase 2 fine-tuning we need real GPU: A100 80GB minimum, ideally H100 80GB. Now that data is laptop-local, RunPod is needed **only for the training runs**, not for storage.

### Steps

1. Create account at https://runpod.io (alternative: https://vast.ai)
2. Add $50 starter credit (Stripe / wire / crypto)
3. Generate SSH key locally, add public key to RunPod settings:
   ```bash
   ssh-keygen -t ed25519 -C "tactical-cz@runpod"
   cat ~/.ssh/id_ed25519.pub  # paste into RunPod → Settings → SSH Public Keys
   ```
4. Create a Pod:
   - **GPU:** A100 80GB (community pricing ~$0.79/hr)
   - **Disk:** **50 GB** persistent volume (just for checkpoints + cached HF model weights; the 19 GB BAS data rides up with the rsync)
   - **Template:** RunPod PyTorch 2.5 + CUDA 12.4 (or newer)
   - **Region:** EU if available (lower latency from Czechia)
5. SSH in, clone the repo, rsync the dataset up from your laptop (~10 min on a good link):
   ```bash
   # On laptop, after SSH'ing into pod once to grab its address:
   rsync -avz --progress data/raw/soccernet/ \
     root@<pod-ip>:/workspace/tactical-cz/data/raw/soccernet/
   ```
6. On the pod:
   ```bash
   git clone https://github.com/sandovabarbora/tactical-cz.git
   cd tactical-cz
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH=$HOME/.local/bin:$PATH
   uv sync --extra train  # torch + lightning + wandb
   uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

### Budget guard

- A100 80GB community pricing ~$0.79/hr (2026)
- Budget cap per the PRD: $450 total
- Plan: fine-tune in focused weekend bursts. Pod stopped when not training = no GPU hourly charge (persistent volume stays at ~$0.10/GB/month).
- Hard alarm at 200 GPU hours.

---

## 3. MatchVision / SoccerMaster availability check

The foundation models the research agents recommended (MatchVision: CVPR'25 arXiv:2412.01820, SoccerMaster: Dec'25 arXiv:2512.11016). They need to be open-weights for us to fine-tune.

### Verification

1. Check HuggingFace: https://huggingface.co/papers/2412.01820 and https://huggingface.co/papers/2512.11016 — look for "Models" tab at the bottom
2. If weights released as `lightonai/matchvision-base` or similar, `transformers.AutoModel.from_pretrained(...)` works directly
3. If only paper + code released, fallback:
   - **V-JEPA 2.1** (https://huggingface.co/facebook/vjepa2-base) as video backbone
   - **VideoMAE v2** (https://huggingface.co/MCG-NJU/videomae-base)
   - Train a custom head for BAS over the frozen backbone (adds 1-2 weeks)

### Decision gate

- IF MatchVision/SoccerMaster open: 3 weeks Phase 2 (fine-tune direct)
- IF only V-JEPA2/VideoMAE2 available: 4-5 weeks Phase 2 (custom BAS head + training)

---

## 4. Events module scaffold

`src/tactical_cz/events/` is **trigger-ready, MPS-only Phase 2 path** (2026-05-18 refactor):
- `downloader.py` ✓ HF download + AES extract via pyzipper
- `dataset.py` ✓ schema-correct skeleton against verified Labels-ball.json layout
- `model.py` ✓ V-JEPA2-L backbone + BAS head, MPS-smoke-tested on a real Sparta clip (0.41s/clip)
- `cache_embeddings.py` ✓ runs V-JEPA2 once over a split, writes embeddings_{split}.parquet (one row per event-centered clip window + sampled negatives)
- `train.py` ✓ head-only loop over cached embeddings (no cv2 in hot path), seconds per epoch, AdamW + cosine, BCEWithLogits with optional inverse-freq pos_weight for rare classes
- `infer.py` ✓ sliding-window inference over any mp4; tolerates either head-only checkpoint or full BASModel checkpoint
- `__main__.py` ✓ CLI shim (prereq checklist)

### Architectural choice: frozen-backbone + cached embeddings

V-JEPA2-L stays FROZEN through Phase 2. Re-encoding 100K clips every epoch is pointless; encode once, train the head over cached vectors. Wins:
- Head training = seconds per epoch (BCE over `(B, 1024) → (B, 12)`)
- Iterating head designs (linear vs MLP, dropout, threshold per class) is free
- Whole pipeline runs on laptop MPS, **zero cloud bill for Phase 2 baseline**
- RunPod GPU becomes optional, only needed if you want to LoRA-finetune the backbone (Phase 3)

### Trigger sequence (post-NDA-password)

```bash
# 0. One-time: set the NDA password (do this in shell rc to persist)
export SOCCERNET_PASSWORD=<from approval email>

# 1. Extract the data you've already downloaded (90 seconds)
uv run python -m tactical_cz.events.downloader extract --split valid

# 2. Encode it once with V-JEPA2-L. MPS: hours per split (overnight for train),
#    minutes for valid. Resumable — flushes parquet every 5 matches.
uv run python -m tactical_cz.events.cache_embeddings --split valid

# 3. Train the head. Seconds per epoch.
uv run python -m tactical_cz.events.train \
    --train-embeddings data/processed/embeddings_train.parquet \
    --val-embeddings   data/processed/embeddings_valid.parquet \
    --epochs 30

# 4. Run end-to-end inference on a Czech broadcast clip
uv run python -m tactical_cz.events.infer \
    --source data/raw/sparta_latest_30s.mp4 \
    --checkpoint checkpoints/head_best.pt \
    --out data/processed/events_timeline.parquet
```

CLI entry: `make events INPUT=data/processed/vision_tracking.parquet`

---

## 5. Phase 2 deliverable

End of Phase 2:

1. **Fine-tuned BAS model** (~500 MB checkpoint on HF Spaces, GPL-3.0) that runs on broadcast video and emits per-event timeline parquet
2. **`data/processed/events_timeline.parquet`** — cols: clip_path, frame_idx, event_type, confidence
3. **Per-event mAP score** vs SoccerNet BAS 2025 leaderboard baseline (target ≥0.50 mAP)
4. **Blog-quality writeup** of training: data prep, hyperparams, validation curve, mistakes, what surprised

---

## Resume instructions

Fresh session start:

```bash
cd /Users/barbora.sandova/Documents/Coding/sandbox/tactical-cz
cat docs/PHASE2_KICKOFF.md     # this file
ls data/raw/soccernet/spotting-ball-2025/   # confirm splits downloaded
ls src/tactical_cz/events/     # see what's been built
```

Phase 2 is now blocked only on:
- [ ] RunPod account + $50 credit + SSH key + first pod (your action; ~15 min)
- [ ] MatchVision / V-JEPA2 backbone availability decision (5 min check on HF)

Once both done, fine-tuning can start.

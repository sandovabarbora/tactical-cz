# Phase 2 Kickoff — Events Layer

**Goal:** action spotting on broadcast video (goal, shot, pass, corner, foul, card, throw-in). Fine-tune MatchVision / SoccerMaster on SoccerNet's Ball Action Spotting (BAS) task; emit per-clip event timeline parquet.

**Timeline:** Phase 2 = weeks 3-5 of the 14-week plan in [PRD.md § Implementation phases](../PRD.md#implementation-phases-14-weeks-part-time).

This doc lists the **prerequisite external actions** (Barbora has to do them herself) before any model fine-tuning can start. Treat as a checklist.

---

## 1. SoccerNet NDA + data access

The 500-game broadcast video corpus is gated by a research-only NDA. The form is online, signing is fast, no academic affiliation required (independent researchers regularly get approved).

### Steps

1. Visit https://www.soccer-net.org/data
2. Click **"Request Access"** (the NDA form)
3. Form asks for: name, affiliation (write *"independent researcher, datasimply.eu"*), purpose (write *"open-source vision baseline for Czech-league broadcast video analysis; non-commercial portfolio project"*)
4. Email goes to soccernet@uliege.be confirming access; reply usually within 24-72 h
5. Once approved, follow their `pip install SoccerNet` workflow and download the relevant task package:

```bash
# Already in our deps
uv run python -c "
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory='data/raw/soccernet')
d.password = '<your NDA password>'  # comes in approval email
d.downloadDataTask('ball-action-spotting-2025', split=['train','valid','test'])
"
```

### What we actually need

For Phase 2 we only need the **Ball Action Spotting (BAS) 2025 task**: ~200 GB of broadcast clips at 25 fps with frame-level event labels (goal, shot, pass, corner, free-kick, throw-in, foul, card, drive, save, etc.). Skip everything else for now (action spotting full corpus, GSR, MVFouls — separate downloads, would balloon to 2 TB).

### Storage planning

- 200 GB of broadcast clips won't fit on most laptop SSDs. Plan to download directly to the RunPod instance (see § 2), not local.
- If working locally with a small subset for prototyping, download 20-30 clips manually and store in `data/raw/soccernet/`.

---

## 2. RunPod (or Vast.ai) account + first A100 instance

Local Mac MPS works for inference (Phase 1 vision baseline). For Phase 2 fine-tuning we need real GPU: A100 80GB minimum, ideally H100 80GB.

### Steps

1. Create account at https://runpod.io (alternative: https://vast.ai for slightly cheaper community A100s)
2. Add $50 starter credit (Stripe / wire / crypto — RunPod takes all three)
3. Create a Pod with these specs:
   - **GPU:** A100 80GB (community pricing ~$0.79/hr)
   - **Disk:** at minimum 300 GB persistent volume (SoccerNet BAS + model checkpoints)
   - **Template:** RunPod PyTorch 2.5 + CUDA 12.4 (or newer)
   - **Region:** EU if available (lower latency from Czechia)
4. SSH key — generate locally with `ssh-keygen -t ed25519 -C "tactical-cz@runpod"` and add the public key to your RunPod settings
5. Start the pod; you'll get an SSH command like `ssh root@123.45.67.89 -p 12345`
6. First-time setup on the pod:

```bash
# Once SSHed in
apt update && apt install -y git tmux htop nvtop
git clone https://github.com/sandovabarbora/tactical-cz.git
cd tactical-cz
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv sync --extra train  # uses extra: torch + lightning + wandb
# Verify GPU
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Budget guard

- A100 80GB community pricing is ~$0.79/hr in 2026
- Budget cap per the PRD: $450 total
- Plan to fine-tune in **focused weekend bursts**, not continuous — pod stopped when not training = no GPU hourly charge (but persistent volume stays at ~$0.10/GB/month)
- Set a hard alarm in your calendar at 200 GPU hours so you don't drift past budget

---

## 3. MatchVision / SoccerMaster availability check

These are the foundation models the research agents recommended (MatchVision: CVPR'25 arXiv:2412.01820, SoccerMaster: Dec'25 arXiv:2512.11016). They need to be open-weights for us to fine-tune.

### Verification

1. Check HuggingFace: https://huggingface.co/papers/2412.01820 and https://huggingface.co/papers/2512.11016 — look for "Models" tab at the bottom
2. If weights are released as `lightonai/matchvision-base` or similar, we can `transformers.AutoModel.from_pretrained(...)` straight in
3. If only paper + code released (no weights), fallback plan:
   - Use **V-JEPA 2.1** (https://huggingface.co/facebook/vjepa2-base) as the video backbone
   - Or **VideoMAE v2** (https://huggingface.co/MCG-NJU/videomae-base)
   - Train a custom head for BAS over those frozen backbones — adds ~1-2 weeks to Phase 2 but is well-trodden ground

### Decision gate

- IF MatchVision/SoccerMaster open: 3 weeks Phase 2 (fine-tune is direct)
- IF only V-JEPA2/VideoMAE2 backbone available: 4-5 weeks Phase 2 (custom BAS head + training)

Don't decide architecture until you've actually checked HF for weights — papers sometimes promise releases that take 6+ months.

---

## 4. Events module scaffold (already in repo)

`src/tactical_cz/events/` has empty `__init__.py`. Phase 2 will add:

```
events/
├── __init__.py
├── downloader.py     SoccerNet BAS package fetch + cache (wraps SN devkit)
├── dataset.py        PyTorch Dataset wrapping SoccerNet clips → frames + labels
├── model.py          MatchVision/SoccerMaster wrapper, OR V-JEPA2 + custom head
├── train.py          Fine-tuning loop (Lightning), wandb logging, checkpoint mgmt
├── infer.py          Run trained model on a tactical-cz vision parquet
├── __main__.py
```

CLI entry: `make events INPUT=data/processed/vision_tracking.parquet`

---

## 5. Phase 2 deliverable

End of Phase 2, you should have:

1. **A fine-tuned BAS model** (~500 MB checkpoint on HF Spaces) that runs on broadcast video and emits per-event timeline parquet
2. **`data/processed/events_timeline.parquet`** — columns: clip_path, frame_idx, event_type, confidence
3. **Per-event mAP score** vs SoccerNet BAS 2025 leaderboard baseline (target ≥0.50 mAP — not winning the benchmark, but solid)
4. **A blog post-quality writeup** of training: data prep, hyperparams, validation curve, mistakes, what surprised

---

## Resume instructions

If you start a fresh session on Phase 2:

```bash
cd /Users/barbora.sandova/Documents/Coding/sandbox/tactical-cz
cat docs/PHASE2_KICKOFF.md     # this file
# Confirm checklist:
# [ ] SoccerNet NDA approved (check email for password)
# [ ] RunPod account + $50 credit + pod created
# [ ] MatchVision availability checked
ls src/tactical_cz/events/     # should be empty stubs
```

Phase 2 starts when both NDA + RunPod are in place. Until then it's blocked on external action.

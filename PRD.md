# tactical-cz — Project Requirements Document

**Last updated:** 2026-05-18
**Author:** Barbora Šandová
**Status:** Phase 0 (scaffolding complete, no ML code yet)

---

## What this is

End-to-end multimodal AI pipeline that consumes a Czech football broadcast (Fortuna Liga / ČFL) and produces a Czech-language tactical analysis. Vision detects players/ball/events, ASR transcribes commentary, an LLM agent answers natural-language tactical questions grounded in both.

**One-line pitch:** *"Vyfoť mi Spartin přesilovkový pressing v 2. poločase proti Slavii"* — broadcast video in, Czech tactical answer out, with video citations.

## Why this exists

Two strategic reasons:

1. **Portfolio diferenciátor.** Generic football data analysis (xG dashboards, radar charts, scouting tools) is saturated. Computer vision + multimodal AI applied to Czech-language broadcast is genuine open territory in 2026. Differentiates from every other "data scientist with football interest" portfolio.

2. **Real commercial gap.** SkillCorner / StatsBomb / Wyscout don't cover Czech Fortuna Liga at the tracking + tactical layer. Smaller clubs (ČFL II, MSFL, Czech II) have no commercial AI option. Tool that runs on broadcast video alone is the only economically viable AI assistant for them.

**Not aspiring to:** replace human analysts, replace SkillCorner for elite clubs, predict CL outcomes. Those are out of scope and structurally infeasible from public data (proven by sister project `sparta-european-context`).

## Audience

**Primary:** Czech football analytical community — sports journalists, ČFL II/MSFL club analysts, football tactical podcasts, academic sports analytics groups (FTVS UK, ČVUT sports labs).

**Secondary:** Recruiters evaluating "data engineer with ML chops + sports domain" for football industry roles (Czech FA, Wyscout, Hudl, SkillCorner localization, mid-sized European clubs hiring remote).

**Tertiary:** General ML/AI community — open-source release on GitHub + HuggingFace Space as portfolio demo URL.

**Explicit non-audience:** AC Sparta Praha and other top-tier clubs with internal Wyscout/SkillCorner data. They have superset of this tool's output. They are the audience for the sister project `sparta-european-context`'s "calibration mirror" framing, not this tool's content.

## Success criteria

In priority order:

1. **Working end-to-end demo at `tactical.datasimply.eu`** (or HF Space). User uploads a Czech-league broadcast clip, gets tactical timeline + Czech-language summary within 5 minutes.

2. **Open-source release on GitHub.** Reproducible from clean clone with documented GPU rental. MIT license.

3. **Pretrained model artifact on HuggingFace.** At minimum a fine-tuned Whisper-Czech-Football ASR; ideally also a fine-tuned MatchVision/SoccerMaster checkpoint on Czech-league examples.

4. **One blog post or workshop paper.** Either at datasimply.eu (project writeup) or submitted to MMSports / CVsports workshop at a sports-ML venue.

5. **At least one Czech sports journalist or analyst engages with the demo** publicly (tweet, podcast mention, article). Validation that the niche is real.

## Out of scope (explicitly)

- Real-time / in-game tactical recommendations (latency unfit for live broadcast use)
- Player injury prediction (no fitness data)
- Transfer-market valuation (sister project covered the Atlas + Scout pattern)
- xG-style probabilistic shot models (saturated literature)
- Multi-language beyond Czech (English/Slovak as future, not v1)
- Mobile app
- Stripe integration / commercial SaaS (until user feedback validates demand)

## Architecture (5 layers)

```
INPUT: Broadcast video (mp4, ≥720p, full match or clip)
   │
   ├─ Layer 1: VISION
   │    Roboflow `sports` + RF-DETR fine-tune
   │    ↓ player + ball detection, tracking, homography → 2D minimap
   │    ↓ per-frame parquet
   │
   ├─ Layer 2: EVENTS
   │    MatchVision / SoccerMaster fine-tune (SoccerNet BAS task)
   │    ↓ action spotting: goal, shot, pass, corner, foul, card, throw-in
   │    ↓ event timeline parquet
   │
   ├─ Layer 3: AUDIO
   │    Whisper-large-v3 baseline → CZ fine-tune
   │    ↓ ASR + speaker diarization (commentator vs co-commentator)
   │    ↓ timestamped transcript parquet
   │
   ├─ Layer 4: MULTIMODAL ALIGNMENT
   │    Join events ↔ transcript timestamps
   │    Tactical pattern detection (formation shifts, pressing triggers,
   │    set-piece taxonomies) from tracking + events
   │    ↓ enriched event timeline with commentary context
   │
   └─ Layer 5: LLM AGENT (Claude 4.7 Sonnet / Llama-3.3-70B)
        Tools:
          - query_events(team, time_range, event_type)
          - retrieve_video_clip(start_ts, end_ts) → returns mp4 segment
          - get_tactical_pattern(pattern_name, team, time_range)
          - summarize_phase(start_ts, end_ts) → natural-language Czech summary
        ↓
OUTPUT: Gradio interface
   - Upload video
   - See: event timeline, tactical pattern detections, minimap replay
   - Ask: Czech-language QA, get answer + video citations
```

## Implementation phases (~14 weeks part-time)

| Phase | Wks | Deliverable | Depends |
|---|---|---|---|
| **0** | 1 | Repo scaffold + PRD + deps + GH Actions + datasimply.eu subdomain setup | — |
| **1** | 2 | Vision L1 baseline runs end-to-end on 1 Czech-league match (Roboflow sports pretrained) | 0 |
| **2** | 3 | Events L2 MatchVision/SoccerMaster fine-tune (RunPod A100, SoccerNet NDA done) | 1 |
| **3** | 2 | Whisper Czech baseline tested + small fine-tune corpus built (50-100 commentary minutes) | 0 |
| **4** | 3 | Multimodal L4 alignment + 3-5 tactical patterns detected (pressing, set-pieces, formation) | 1, 2, 3 |
| **5** | 1-2 | LLM agent L5 with Claude tool-use; Czech QA functional | 4 |
| **6** | 1 | Gradio demo + HF Space + tactical.datasimply.eu deployed | 5 |
| **7** | 1 | Blog post writeup + Rosický cover letter v2 + Twitter/LinkedIn share | 6 |

**Total elapsed:** ~14 weeks at evenings + weekends (10-15 hrs/week) = 3.5 months.

## Data sources (all public, all legal)

- **SoccerNet** (NDA-but-not-affiliation-gated, request via web form): broadcast clips + annotations for fine-tuning vision models
- **Kaggle DFL Bundesliga Data Shootout 2022**: pretrained baseline reference
- **StatsBomb open data**: event-data examples for sanity-checking event detection output
- **YouTube — Fortuna Liga / O2 TV Sport / club channels**: legitimately published highlight reels for Czech-language inference test set
- **SoccerNet-Echoes** (Apache 2.0): multilingual commentary template; we extend to Czech
- **HuggingFace pretrained**: Whisper-large-v3, MatchVision (if open-weight), V-JEPA2 / VideoMAE2 (fallback backbones)

## Compute budget

| Item | Estimate |
|---|---|
| RunPod A100 80GB community @ $0.79/hr × 150 hrs (MatchVision FT, Whisper FT, ablations) | $120 |
| H100 × 20 hrs for final FT runs ($1.55/hr Vast.ai) | $30 |
| HuggingFace Space hosting (CPU upgraded → GPU on inference) | $50/month × 4 months = $200 |
| Buffer for unexpected reruns + larger ablation | $100 |
| **Total** | **~$450** (within $500 cap) |

## Tech stack

- **Python 3.12** (uv-managed)
- **PyTorch + HuggingFace Transformers + Lightning** (model training)
- **Ultralytics YOLO + BoxMOT + supervision** (vision baseline via Roboflow sports)
- **SoccerNet devkit** (`pip install SoccerNet`) for data + benchmark integration
- **OpenAI Whisper / faster-whisper** (ASR)
- **mplsoccer + kloppy + socceraction** (football-specific analytics)
- **Gradio** (demo interface)
- **HuggingFace Spaces** (deployment)
- **Anthropic Claude API** (LLM agent — Sonnet 4.6 for cost, Sonnet 4.7 for production)

## Risks + mitigations

1. **MatchVision / SoccerMaster weights might not be open** — Mitigation: V-JEPA2 + custom head; lose some headroom but stays open-source path
2. **Czech commentary corpus is small** — Mitigation: start with multilingual Whisper baseline (handles Czech reasonably); fine-tune only if domain-specific gaps surface
3. **Czech broadcast copyright** — Mitigation: use only legitimately-published highlight reels (club YouTube channels, O2 TV official clips); avoid full-match recordings except SoccerNet
4. **GPU costs creep** — Mitigation: hard cap at $500, fail loudly if any single training run goes over $50
5. **Skill ramp on multimodal alignment** — Mitigation: lean heavily on existing HF tools (e.g. `transformers` cross-modal helpers), avoid bespoke architecture work
6. **Project drift / scope creep** — Mitigation: this PRD is the source of truth; weekly self-check against "phase deliverables" only

## Brand context (for impeccable + future sessions)

- **Domain:** `tactical.datasimply.eu` subdomain (matches sister projects: color-fingerprint-cz, sound-fingerprint-cz, strategic-topography-cz, spot-scorer all under datasimply.eu)
- **Visual register:** TBD per `/impeccable teach` in Phase 6. Will not blindly inherit `sparta-european-context` Sparta-brand pivot — this is a generic Czech football tool, not Sparta-specific.
- **Voice:** Czech-language primary content, English code comments + commit messages, README in both
- **Licence:** MIT throughout

## How to resume from this PRD

If session ends here and you restart from scratch:

```bash
cd /Users/barbora.sandova/Documents/Coding/sandbox/tactical-cz
cat PRD.md            # this file
cat README.md         # public face
cat PRODUCT.md        # impeccable-format strategic context (Phase 6+)
make help             # what's wired
git log --oneline     # what's shipped
```

Phase 1 entry point will be `src/tactical_cz/vision/` once we wire it.

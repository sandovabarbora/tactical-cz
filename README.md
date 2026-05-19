---
title: tactical-cz
emoji: ⚽
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
short_description: Multimodální AI analýza českého fotbalu — Q&A nad pipeline
---

# tactical-cz

> Multimodální AI pipeline pro česká fotbalová televizní vysílání. Vize, ASR, taktický LLM agent.

**Stav:** Phase 0-4 hotovo. Statický demo report na [GitHub Pages](https://sandovabarbora.github.io/tactical-cz/), interaktivní Q&A demo přes Gradio (`app.py` na HF Spaces). Viz [PRD.md](./PRD.md) pro celý projektový plán.

## Demos

| | URL | Co to ukazuje |
|---|---|---|
| 📊 Static report | https://sandovabarbora.github.io/tactical-cz/ | Annotated mp4 + 4-vrstvý pipeline analysis + diagnostické findings + pre-computed Q&A |
| 💬 Interactive Q&A | (HF Space link po deploy) | Ptej se na konkrétní moment v Sparta-goals klipu, Czech default |

---

## Co to bude

End-to-end pipeline který vezme TV vysílání zápasu Fortuna Ligy a vrátí česky psanou taktickou analýzu:

```
Broadcast video (mp4)
   │
   ├─ VISION    detekce hráčů + míče + hřiště, tracking, minimapa
   ├─ EVENTS    rozpoznávání akcí (gol, střela, faul, roh, karta)
   ├─ AUDIO     česká ASR komentáře (Whisper fine-tune)
   ├─ ALIGN     spojení event timeline + transkript + taktické vzorce
   └─ AGENT     česká QA: "Kde Sparta v 2. poločase tlačila výš?"
        ↓
Gradio demo + taktický report v češtině s video citacemi
```

## Proč

Žádný komerční produkt nepokrývá Fortuna Ligu na trackingové úrovni. SkillCorner / StatsBomb / Wyscout cílí top-tier ligy. Pro menší české kluby (ČFL II, MSFL, divize) je *broadcast video jejich jediná data*. Tahle pipeline je první open-source pokus to změnit.

Druhotně: portfolio piece pro AI/ML role v sportovním analytics prostoru.

## Tech stack

Python 3.12 (uv) · PyTorch + HuggingFace · Ultralytics YOLO + Roboflow `sports` · SoccerNet devkit · Whisper / faster-whisper · Anthropic Claude API · Gradio · HuggingFace Spaces

## Roadmap (14 týdnů part-time)

| Phase | Týdny | Co |
|---|---|---|
| 0 | 1 | Scaffolding + PRD ✅ |
| 1 | 2 | Vision baseline (Roboflow sports) na 1 zápasu |
| 2 | 3 | Events fine-tune (MatchVision / SoccerMaster na SoccerNet BAS) |
| 3 | 2 | Whisper CZ fine-tune |
| 4 | 3 | Multimodální alignment + 3-5 taktických vzorců |
| 5 | 1-2 | LLM agent s tool-use, česká QA |
| 6 | 1 | Gradio demo + `tactical.datasimply.eu` deploy |
| 7 | 1 | Blog post + Rosický cover letter v2 |

Detaily v [PRD.md](./PRD.md).

## Quick start (pro budoucí mě)

```bash
make install-dev    # uv sync s dev extras
make help           # všechny dostupné cíle
make test           # offline testy
```

Phase-konkrétní entry pointy jsou v [PRD.md § Implementation phases](./PRD.md#implementation-phases-14-weeks-part-time).

## Sourozenecké projekty pod datasimply.eu

- [`color-fingerprint-cz`](https://sandovabarbora.github.io/color-fingerprint-cz/) — visual analysis Czech ad spots
- [`sound-fingerprint-cz`](https://sandovabarbora.github.io/sound-fingerprint-cz/) — audio analysis
- [`strategic-topography-cz`](https://sandovabarbora.github.io/strategic-topography-cz/) — semantic homepage analysis
- [`spot-scorer`](https://sandovabarbora.github.io/spot-scorer/) — interactive spot scorer (Modal backend)
- [`sparta-european-context`](https://sandovabarbora.github.io/sparta-european-context/) — methodological positioning of Sparta Praha (sister project to this one)
- **tactical-cz** *(this)* — multimodální AI pro CzFL broadcasty

## Licence

MIT (kód) + Apache 2.0 / CC-BY-NC pro derivative datasety (per SoccerNet a další upstream).

## Autor

**Barbora Šandová** · Data &amp; Cloud Engineer at Alma Career  
[datasimply.eu](https://datasimply.eu) · [LinkedIn](https://www.linkedin.com/in/barborasandova) · [GitHub](https://github.com/sandovabarbora)

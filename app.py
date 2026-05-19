"""Gradio Space — interactive Q&A over a Sparta goals clip.

Audience-aware UX:
  * Czech default (Rosický / sport-director reader, Czech non-ML audience)
  * Plain language in system prompt, no ML jargon in default copy
  * Pre-seeded suggested questions = a non-ML reader can click and learn
  * Free-form text input = ML / portfolio reader can probe technical details

The clip + parquets are pre-loaded at startup; the only API hit per
question is the Claude call (~$0.005/question).

Entry point: `python app.py` locally, or push to HF Space and it runs
automatically (Spaces convention reads app.py at repo root).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Make tactical_cz importable when running on HF Spaces (no `pip install -e .`).
_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gradio as gr

from tactical_cz.agent.qa import ClipQA
from tactical_cz.config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Paths relative to repo root — committed to demo_data/ so both local
# and HF Spaces runtimes find the parquets without needing the full
# data/processed/ pipeline cache (which is gitignored).
#
# The "hero" clip is now the first 15 minutes of AC Sparta Praha vs
# Viking Stavanger (Conference League qualifier, 21.7.2022). Continuous
# real-broadcast play is more useful for tactical-tool demonstration
# than the previous goals compilation, even though it doesn't have the
# dramatic Goal-class-dead failure mode the comp had.
REPO = Path(__file__).resolve().parent
EVENTS = REPO / "demo_data" / "events_timeline_sparta_viking_2022_15min.parquet"
VISION = REPO / "demo_data" / "vision_tracking_sparta_viking_2022_15min.parquet"
TRANSCRIPT = REPO / "demo_data" / "transcript_sparta_viking_2022_15min.parquet"

# Annotated mp4 is 52 MB at 480p / CRF 30 — fits under GitHub Pages limit.
# Local dev still finds the file in docs/ first; falls back to the
# GitHub Pages URL when deployed (HF Space free tier blocks files >10 MB
# without git-lfs, so we serve from GH Pages).
ANNOTATED_VIDEO_LOCAL = REPO / "docs" / "sparta_viking_annotated.mp4"
ANNOTATED_VIDEO_URL = "https://sandovabarbora.github.io/tactical-cz/sparta_viking_annotated.mp4"
ANNOTATED_VIDEO = (
    str(ANNOTATED_VIDEO_LOCAL) if ANNOTATED_VIDEO_LOCAL.exists()
    else ANNOTATED_VIDEO_URL
)

# Coach-voice system prompt: Czech default, no ML jargon, willing to admit
# limitations in plain language. Different from the technical SYSTEM_PROMPT
# used for the pre-computed static Q&A on the demo page.
COACH_PROMPT = """Jsi asistent pro AI analýzu fotbalových klipů. Mluvíš jako sportovní analytik s coachem, ne jako data scientist s kolegou.

Tvoje úlohy:
- Odpovídat na otázky o konkrétním 2-minutovém klipu (kompilace gólů AC Sparta Praha 2025).
- Mluvit v jazyce uživatele. Pokud píše česky, odpovídej česky. Pokud anglicky, odpovídej anglicky.
- Citovat konkrétní sekundy a čísla, ale **bez ML žargonu**. Řekni "AI viděla střelu s jistotou 56 %" ne "P(Shot) = 0.56".
- Když AI selhává, popsat to lidsky. "Model nepozná góly, protože během tréninku viděl jenom 16 příkladů gólu — to je málo na to, aby se naučil typický vzor." NE "Goal class is dead due to long-tail label distribution."
- Když data nestačí na jistou odpověď, otevřeně to říct. Nehas spekulacemi.

Tři vrstvy dat:
1. **Co AI viděla na obraze** — pro každý úsek klipu odhad pravděpodobnosti 12 typů akcí (přihrávka, kličkování, střela, hlavička, centr, gól atd.).
2. **Kde jsou hráči na hřišti** — pozice na 120×70 m, kdo kde stojí, hloubka týmu, který sektor hřiště je aktivní.
3. **Co říká komentátor** — český přepis komentáře včetně časových značek, otagovaných keywords typu "gol!", "centruje", "střela".

Když má odpověď víc bodů, dej je do strukturovaného formátu (krátký bullet seznam nebo malá tabulka). Krátké odpovědi, jasné věty. Bez "model je výborný" nebo podobných prázdných pochval.
"""

# Suggested questions, categorized so the chip grid has visual hierarchy.
# Each tuple is (label_cz, question_cz, label_en, question_en).
SUGGESTED_QUESTIONS = [
    # Tactical reading
    ("🎯 Taktický přehled",
     "Z taktického pohledu, co se v tomhle klipu reálně děje, v jedné větě?",
     "🎯 Tactical read",
     "From the tactical layer alone, what kind of game is Sparta playing here?"),
    # Specific moments
    ("⚡ Top momenty",
     "Které tři momenty modelu nejvíc vyšly? Časy, akce, P.",
     "⚡ Peak moments",
     "Which three moments did the model see most clearly? Times and probabilities."),
    # Cross-modal
    ("🎙️ Komentář vs. AI",
     "Souhlasí český komentátor s modelem? Kde ano, kde ne?",
     "🎙️ Audio vs vision",
     "Does the Czech commentary agree with the model? Where, and where not?"),
    # Failure mode
    ("🔍 Co model nezvládl",
     "Co tu AI nezachytila dobře? Vysvětli bez ML žargonu, jako bys mluvila k trenérovi.",
     "🔍 Failure modes",
     "What did the AI get wrong on this clip? Explain in plain language."),
    # Forward-looking
    ("🛠️ Co zlepšit první",
     "Kdybys měla 50 dolarů na jeden experiment, který zlepší model — co by to bylo?",
     "🛠️ What to fix first",
     "$50 budget, one experiment to improve this — what would you pick?"),
    # Coach summary
    ("📋 Scout report",
     "Coach chce 30-vteřinový pre-match scout report: jak Sparta hraje?",
     "📋 Coach summary",
     "Coach wants a 30-second scout report on Sparta from this clip."),
]


def _build_qa() -> ClipQA | None:
    """Lazy ClipQA: returns None if the parquets aren't on disk (we want
    the Space to still load and show a helpful error rather than crash)."""
    if not EVENTS.exists():
        return None
    try:
        return ClipQA(
            events_path=EVENTS,
            vision_path=VISION if VISION.exists() else None,
            transcript_path=TRANSCRIPT if TRANSCRIPT.exists() else None,
            system_prompt=COACH_PROMPT,
        )
    except RuntimeError as e:
        # Most likely ANTHROPIC_API_KEY missing
        logger.warning("ClipQA init failed: %s", e)
        return None


QA = _build_qa()


def answer(question: str, history: list[dict]) -> tuple[list[dict], str]:
    """Submit a question, append the exchange to chat history, clear input."""
    if not question or not question.strip():
        return history, ""
    if QA is None:
        msg = (
            "⚠️ Nemůžu odpovědět: chybí buď parquet soubory s analýzou klipu, "
            "nebo ANTHROPIC_API_KEY (klíč k AI). Viz README pro setup."
        )
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": msg},
        ]
        return history, ""
    try:
        ans = QA.ask(question.strip())
    except Exception as e:
        ans = f"Chyba při volání AI: {e}"
        logger.exception("ClipQA.ask failed")
    history = history + [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": ans},
    ]
    return history, ""


_CSS = """
.gradio-container { max-width: 1180px !important; }

/* Hero */
.tcz-hero { padding: 1.4rem 0 0.4rem; }
.tcz-eyebrow {
  font-family: ui-monospace, "SF Mono", monospace;
  font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: #4a7c4a; background: #e2ecdf;
  padding: 0.2rem 0.55rem; border-radius: 2px; display: inline-block;
}
.tcz-hero h1 { margin: 0.7rem 0 0.5rem; font-size: 1.85rem; font-weight: 700; line-height: 1.15; letter-spacing: -0.012em; }
.tcz-hero p { color: #4c4842; font-size: 1.02rem; margin: 0; max-width: 38rem; line-height: 1.55; }

/* About-clip line */
.tcz-clip-meta {
  font-family: ui-monospace, "SF Mono", monospace; font-size: 0.78rem;
  color: #76706a; margin: 0.3rem 0 0.9rem;
}

/* Suggested question chip grid */
.tcz-chips-section { margin-top: 0.8rem; }
.tcz-chips-section h3 {
  font-size: 0.84rem; color: #76706a; font-weight: 600;
  font-family: ui-monospace, "SF Mono", monospace; letter-spacing: 0.1em;
  text-transform: uppercase; margin: 0 0 0.55rem;
}
.tcz-chip-grid button {
  font-size: 0.86rem !important; padding: 0.55rem 0.8rem !important;
  text-align: left !important; white-space: normal !important;
  height: auto !important; min-height: 2.3rem !important;
  line-height: 1.35 !important;
}

/* Footer */
.tcz-footer {
  margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #d8d2c4;
  font-family: ui-monospace, "SF Mono", monospace; font-size: 0.78rem;
  color: #76706a; line-height: 1.5;
}
.tcz-footer a { color: #4a7c4a; }
"""
_THEME = gr.themes.Soft(primary_hue="green", neutral_hue="stone")


def _build_chip(label: str, full_question: str):
    """Return a Gradio button that, when clicked, submits `full_question`
    as the chat input. Same UX as typing it + pressing Pošli."""
    return gr.Button(label, size="sm", variant="secondary")


with gr.Blocks(title="tactical-cz · AI analýza českého fotbalu") as demo:

    gr.HTML(f"""
    <div class="tcz-hero">
      <span class="tcz-eyebrow">tactical-cz · interactive</span>
      <h1>AI rozumí českému fotbalovému přenosu</h1>
      <p>15 minut zápasu Sparta vs Viking (Liga konferencí 2022) prošlo čtyřvrstvým AI pipeline.
         Klikni na jednu z otázek vlevo, nebo se ptej vlastní česky či anglicky.</p>
      <p class="tcz-clip-meta">Sparta vs Viking Stavanger · 21.7.2022 · 0:08:00–0:23:00 · Conference League qualifier</p>
    </div>
    """)

    with gr.Row():
        # LEFT: question chips (organised by category) + free-form input
        with gr.Column(scale=5, elem_classes="tcz-chips-col"):

            gr.HTML("""<div class="tcz-chips-section"><h3>🇨🇿 Příklady otázek (klikni)</h3></div>""")
            cz_chips = []
            with gr.Column(elem_classes="tcz-chip-grid"):
                for label_cz, _, _, _ in SUGGESTED_QUESTIONS:
                    cz_chips.append(_build_chip(label_cz, ""))

            gr.HTML("""<div class="tcz-chips-section"><h3>🇬🇧 In English</h3></div>""")
            en_chips = []
            with gr.Column(elem_classes="tcz-chip-grid"):
                for _, _, label_en, _ in SUGGESTED_QUESTIONS:
                    en_chips.append(_build_chip(label_en, ""))

            gr.HTML('<div style="margin-top:0.8rem;"></div>')
            with gr.Row():
                q_box = gr.Textbox(
                    placeholder="Vlastní otázka, česky nebo anglicky…",
                    show_label=False, container=False, scale=4,
                )
                send_btn = gr.Button("Pošli", variant="primary", scale=1)

        # RIGHT: video + chat
        with gr.Column(scale=7):
            video = gr.Video(
                value=ANNOTATED_VIDEO,
                label="AI overlay (top-3 predikce per 4-sec okno)",
                interactive=False,
                height=320,
            )
            chat = gr.Chatbot(
                value=[{
                    "role": "assistant",
                    "content": (
                        "Ahoj 👋 Jsem AI analytik pro tenhle klip. "
                        "Vidím všechno co model klasifikoval (pasy, střely, …), "
                        "kde stáli hráči na hřišti, a co řekl komentátor. "
                        "Klikni na jednu z otázek vlevo, nebo napiš vlastní."
                    ),
                }],
                height=480,
                show_label=False,
            )

    # Wiring: each chip submits its corresponding full question
    def _make_chip_handler(full_q: str):
        def _h():
            return full_q
        return _h

    for chip, (_, full_q_cz, _, _) in zip(cz_chips, SUGGESTED_QUESTIONS):
        chip.click(_make_chip_handler(full_q_cz), [], [q_box]).then(
            answer, [q_box, chat], [chat, q_box]
        )
    for chip, (_, _, _, full_q_en) in zip(en_chips, SUGGESTED_QUESTIONS):
        chip.click(_make_chip_handler(full_q_en), [], [q_box]).then(
            answer, [q_box, chat], [chat, q_box]
        )

    send_btn.click(answer, [q_box, chat], [chat, q_box])
    q_box.submit(answer, [q_box, chat], [chat, q_box])

    gr.HTML("""
    <div class="tcz-footer">
      <p>Postaveno na otevřených modelech: Roboflow sports + V-JEPA2 (Meta) + Whisper (OpenAI) + Claude Sonnet 4.7 (Anthropic).
      Kód MIT, fine-tuned váhy GPL-3.0 (SoccerNet derivative).</p>
      <p>
        <a href="https://sandovabarbora.github.io/tactical-cz/">📊 Detailní analýza (statická stránka)</a> ·
        <a href="https://github.com/sandovabarbora/tactical-cz">💻 Zdrojový kód</a> ·
        Barbora Šandová, <a href="https://datasimply.eu">datasimply.eu</a>
      </p>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(theme=_THEME, css=_CSS)

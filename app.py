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
REPO = Path(__file__).resolve().parent
EVENTS = REPO / "demo_data" / "events_timeline_goals.parquet"
VISION = REPO / "demo_data" / "vision_tracking_goals.parquet"
TRANSCRIPT = REPO / "demo_data" / "transcript_goals.parquet"

# The 37 MB annotated mp4 lives on GitHub Pages (HF Space free tier
# blocks files >10 MB without git-lfs). Local dev still finds the file
# in docs/ first; falls back to the GitHub Pages URL on deployment.
ANNOTATED_VIDEO_LOCAL = REPO / "docs" / "sparta_goals_annotated.mp4"
ANNOTATED_VIDEO_URL = "https://sandovabarbora.github.io/tactical-cz/sparta_goals_annotated.mp4"
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

SUGGESTED_QUESTIONS_CZ = [
    "Co se v tomhle klipu reálně děje, z taktického pohledu, v jedné větě?",
    "Kdy a kde model nejlépe pozná střely?",
    "Komentátor říká 'gol!' šestkrát. Co na to říká AI?",
    "Co nejvíc rozumí AI špatně a proč?",
    "Kdybys ten model měla zlepšit, co bys udělala jako první?",
    "Která Sparta zóna na hřišti je v klipu nejaktivnější?",
]

SUGGESTED_QUESTIONS_EN = [
    "In one sentence, what is happening in this clip tactically?",
    "When does the model see shots most clearly?",
    "The commentator yells 'gol!' six times — what does the AI think?",
    "What does the AI get most wrong, and why?",
    "If you had to fix one thing about this model, what?",
    "Which zone of the pitch is Sparta most active in?",
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
.gradio-container { max-width: 1100px !important; }
.hero { padding: 1.2rem 0; }
.hero h1 { margin: 0 0 0.4rem; font-size: 1.7rem; }
.hero p { color: #4c4842; font-size: 1rem; margin: 0; }
.chip-row button { font-size: 0.85rem !important; padding: 0.4rem 0.7rem !important; }
"""
_THEME = gr.themes.Soft(primary_hue="green", neutral_hue="stone")


with gr.Blocks(title="tactical-cz · Multimodální AI analýza českého fotbalu") as demo:
    gr.HTML("""
    <div class="hero">
      <h1>tactical-cz · Multimodální AI analýza českého fotbalu</h1>
      <p>2-minutový sestřih gólů AC Sparty 2025 prošel přes čtyřvrstvý AI pipeline: rozpoznávání hráčů na hřišti, klasifikace akcí (pasy, střely, hlavičky), přepis českého komentáře, a nahoře nad tím vším Claude jako analytik. Ptej se na konkrétní moment, nebo klikni na jednu z otázek dole.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(
                value=ANNOTATED_VIDEO,
                label="Klip s AI overlay (top-3 predikce per okno, 4-second windows)",
                interactive=False,
            )
            gr.Markdown(
                "*Pravý horní roh klipu (MODEL SEES) ukazuje 3 nejjistější AI predikce. "
                "Pravý dolní chip ukazuje, jak jistá AI je. "
                "Když je 'uncertain', AI tipuje a nemá silný signál.*"
            )

        with gr.Column(scale=1):
            chat = gr.Chatbot(
                value=[{
                    "role": "assistant",
                    "content": (
                        "Ahoj 👋 Jsem AI analytik pro tenhle klip. "
                        "Vidím všechno co model klasifikoval (pasy, střely, góly…), "
                        "kde stáli hráči na hřišti, a co řekl komentátor. "
                        "Klikni na jednu z otázek dole, nebo napiš vlastní."
                    ),
                }],
                height=380,
            )

            with gr.Row():
                q_box = gr.Textbox(
                    placeholder="Napiš vlastní otázku (česky nebo anglicky)…",
                    show_label=False,
                    container=False,
                    scale=4,
                )
                send_btn = gr.Button("Pošli", variant="primary", scale=1)

    gr.Markdown("### 🇨🇿 Příklady otázek (klikni)")
    with gr.Row(elem_classes="chip-row"):
        cz_btns = [gr.Button(q, size="sm", variant="secondary") for q in SUGGESTED_QUESTIONS_CZ]
    gr.Markdown("### 🇬🇧 In English")
    with gr.Row(elem_classes="chip-row"):
        en_btns = [gr.Button(q, size="sm", variant="secondary") for q in SUGGESTED_QUESTIONS_EN]

    # Wiring: button click fills the textbox + immediately submits
    def _click_btn(q):
        return q
    for btn, q in zip(cz_btns, SUGGESTED_QUESTIONS_CZ):
        btn.click(_click_btn, [gr.State(q)], [q_box]).then(
            answer, [q_box, chat], [chat, q_box]
        )
    for btn, q in zip(en_btns, SUGGESTED_QUESTIONS_EN):
        btn.click(_click_btn, [gr.State(q)], [q_box]).then(
            answer, [q_box, chat], [chat, q_box]
        )

    send_btn.click(answer, [q_box, chat], [chat, q_box])
    q_box.submit(answer, [q_box, chat], [chat, q_box])

    gr.HTML("""
    <hr style="margin-top: 2rem; border: 0; border-top: 1px solid #d8d2c4;">
    <p style="font-size: 0.8rem; color: #76706a; text-align: center; margin-top: 1rem;">
      Postaveno na MIT / GPL-3.0 z otevřených modelů: Roboflow sports + V-JEPA2 (Meta) + Whisper (OpenAI) + Claude (Anthropic).<br>
      <a href="https://github.com/sandovabarbora/tactical-cz" style="color: #4a7c4a;">Zdrojový kód a celý report</a> ·
      <a href="https://sandovabarbora.github.io/tactical-cz/" style="color: #4a7c4a;">Detailní analýza</a> ·
      Barbora Šandová, <a href="https://datasimply.eu" style="color: #4a7c4a;">datasimply.eu</a>
    </p>
    """)


if __name__ == "__main__":
    demo.launch(theme=_THEME, css=_CSS)

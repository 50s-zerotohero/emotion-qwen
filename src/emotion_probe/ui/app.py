"""Gradio UI: real-time emotion visualization during Qwen3 generation.

Layout:
  - Top:   Reasoning mode radio (no_think / think / scratchpad)
  - Left:  gr.Chatbot — section-colored HTML tokens
  - Right: two stacked gr.HTML panels
      • Top:    Emotion at ":" token (first generated token, fixed)
      • Bottom: Live emotion (current token, real-time delta from baseline)
  - Bottom: gr.Plot — token × emotion heatmap (shown once after generation)
"""

import asyncio
import sys
from pathlib import Path

import gradio as gr
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emotion_probe.config import load_config
from emotion_probe.backend.base import ReasoningMode
from emotion_probe.backend.local_nnsight import LocalNNSightBackend
from emotion_probe.ui.components import (
    color_token, render_emotion_bars, render_dual_emotion_bars, render_heatmap,
    EMOTIONS,
)

# --------------------------------------------------------------------------- #
# Global state (loaded once at startup)
# --------------------------------------------------------------------------- #
_backend: LocalNNSightBackend | None = None
_emotion_vectors: dict[str, torch.Tensor] | None = None


def _load_globals():
    global _backend, _emotion_vectors
    if _backend is None:
        cfg = load_config()
        _backend = LocalNNSightBackend(cfg)
        _backend._ensure_loaded()   # load model now so first request isn't slow

    if _emotion_vectors is None:
        vec_path = PROJECT_ROOT / "data" / "emotion_vectors.pt"
        _emotion_vectors = torch.load(vec_path, weights_only=True)


# --------------------------------------------------------------------------- #
# Respond generator
# --------------------------------------------------------------------------- #
async def respond(
    message: str,
    history: list,
    mode_str: str,
    system_prompt: str,
    steering_emotion_str: str,
    steering_alpha: float,
):
    """Async generator: yields (history, dual_bars_html, heatmap) per token."""
    if not message.strip():
        yield history, render_dual_emotion_bars({}, {}), None
        return

    mode = ReasoningMode(mode_str)
    cfg  = load_config()
    max_new_tokens: int = cfg["reasoning"]["max_new_tokens"]

    # Gradio 6.x messages format: list of {"role": ..., "content": ...}
    history = list(history) + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": ""},
    ]

    token_records  = []
    response_html  = ""
    colon_scores:   dict[str, float] = {}
    display_scores: dict[str, float] = {}
    baseline_scores: dict[str, float] | None = None

    # Reset emotion panels before generation starts
    yield history, render_dual_emotion_bars({}, {}), gr.update()

    steering_em = None if steering_emotion_str == "None" else steering_emotion_str

    async for token_data in _backend.generate_with_emotions(
        user_message=message,
        mode=mode,
        emotion_vectors=_emotion_vectors,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
        steering_emotion=steering_em,
        steering_alpha=steering_alpha,
    ):
        raw_scores = token_data["emotions"]

        # First token: record raw scores as colon panel and set as baseline
        if baseline_scores is None:
            baseline_scores = raw_scores
            colon_scores    = raw_scores
            display_scores  = {e: 0.0 for e in raw_scores}
        else:
            display_scores  = {e: raw_scores[e] - baseline_scores[e] for e in raw_scores}

        # Store delta scores in the record (used by heatmap)
        delta_record = dict(token_data)
        delta_record["emotions"] = display_scores
        token_records.append(delta_record)

        response_html += color_token(token_data["token"], token_data["section"])
        history[-1] = {"role": "assistant", "content": response_html}

        dual_html = render_dual_emotion_bars(colon_scores, display_scores)
        yield history, dual_html, gr.update()

    # Generation complete → render heatmap (":" column = raw, rest = delta)
    heatmap   = render_heatmap(token_records, colon_scores=colon_scores)
    dual_html = render_dual_emotion_bars(colon_scores, display_scores)
    yield history, dual_html, gr.update(value=heatmap)


# --------------------------------------------------------------------------- #
# Build Gradio app
# --------------------------------------------------------------------------- #
def build_app() -> gr.Blocks:
    with gr.Blocks(title="Emotion Probe — Qwen3-4B") as demo:
        gr.Markdown(
            "# Emotion Probe for Qwen3-4B\n"
            "Real-time emotion activation visualization · "
            "<span style='color:#5a2d82'>■ think</span> "
            "<span style='color:#b45309'>■ scratchpad</span> "
            "■ response"
        )

        mode_radio = gr.Radio(
            choices=["no_think", "think", "scratchpad"],
            value="no_think",
            label="Reasoning Mode",
        )

        system_prompt_box = gr.Textbox(
            label="System Prompt (optional)",
            placeholder="You are a helpful assistant...",
            lines=2,
            value="",
        )

        with gr.Row():
            steering_emotion_dd = gr.Dropdown(
                choices=["None"] + EMOTIONS,
                value="None",
                label="Steering emotion",
            )
            steering_alpha_sl = gr.Slider(
                minimum=-10,
                maximum=10,
                value=0,
                step=0.5,
                label="Steering strength (α)",
                info="Recommended ±1–5. Internally scales by hidden norm (eff = α × norm / 10).",
            )

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Type a message and press Enter or click Send…",
                label="Message",
                scale=4,
                lines=1,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
            stop_btn = gr.Button("Stop", variant="stop", scale=1)

        with gr.Row():
            with gr.Column(scale=3):
                # Gradio 6.x: messages format only, sanitize_html=False to allow spans
                chatbot = gr.Chatbot(
                    label="Response",
                    height=480,
                    render_markdown=True,
                    sanitize_html=False,    # allow <span style=...> for section coloring
                )

            with gr.Column(scale=2):
                emotion_display = gr.HTML(
                    value=render_dual_emotion_bars({}, {}),
                )

        heatmap_display = gr.Plot(label="Token-level Emotion Heatmap")

        # ---- Wire up events ---- #
        # Note: do NOT clear heatmap on msg_box.change — the .then() that clears
        # msg_box after submission would trigger it and erase the just-rendered heatmap.
        send_kwargs = dict(
            fn=respond,
            inputs=[msg_box, chatbot, mode_radio, system_prompt_box,
                    steering_emotion_dd, steering_alpha_sl],
            outputs=[chatbot, emotion_display, heatmap_display],
        )

        # capture the generation events BEFORE chaining .then()
        # so cancels= points to the generator, not the lambda
        submit_gen = msg_box.submit(**send_kwargs)
        submit_gen.then(lambda: "", outputs=[msg_box])

        click_gen = send_btn.click(**send_kwargs)
        click_gen.then(lambda: "", outputs=[msg_box])

        stop_btn.click(fn=None, cancels=[submit_gen, click_gen])

    return demo


def launch(share: bool = False, server_port: int = 7860):
    print("Loading model and emotion vectors…")
    _load_globals()
    print("Ready.\n")

    app = build_app()
    app.queue()
    app.launch(share=share, server_port=server_port)


if __name__ == "__main__":
    launch()

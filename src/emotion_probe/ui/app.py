"""Gradio UI: real-time emotion visualization during Qwen3 generation.

Layout:
  - Top:   Reasoning mode radio (no_think / think / scratchpad)
  - Left:  gr.Chatbot — section-colored HTML tokens
           [Fallback: if HTML rendering is broken in gr.Chatbot, replace with
            gr.HTML(elem_id="chat_display") and manage history manually as HTML]
  - Right: gr.HTML — emotion bar graph (real-time per-token update)
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
from emotion_probe.ui.components import color_token, render_emotion_bars, render_heatmap

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
async def respond(message: str, history: list, mode_str: str):
    """Async generator: yields (history, bars_html, heatmap_or_None) per token."""
    if not message.strip():
        yield history, render_emotion_bars({}), None
        return

    mode = ReasoningMode(mode_str)
    cfg  = load_config()
    max_new_tokens: int = cfg["reasoning"]["max_new_tokens"]

    # Gradio 6.x messages format: list of {"role": ..., "content": ...}
    history = list(history) + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": ""},
    ]

    token_records   = []
    response_html   = ""
    bars_html       = render_emotion_bars({})
    baseline_scores: dict[str, float] | None = None   # first-token scores as baseline

    async for token_data in _backend.generate_with_emotions(
        user_message=message,
        mode=mode,
        emotion_vectors=_emotion_vectors,
        max_new_tokens=max_new_tokens,
    ):
        raw_scores = token_data["emotions"]

        # Establish baseline from first token; subsequent tokens show delta
        if baseline_scores is None:
            baseline_scores = raw_scores
            display_scores  = {e: 0.0 for e in raw_scores}
        else:
            display_scores  = {e: raw_scores[e] - baseline_scores[e] for e in raw_scores}

        # Store delta scores in the record (used by heatmap)
        delta_record = dict(token_data)
        delta_record["emotions"] = display_scores
        token_records.append(delta_record)

        response_html += color_token(token_data["token"], token_data["section"])
        bars_html      = render_emotion_bars(display_scores)

        # Update the last assistant message in place (streaming)
        history[-1] = {"role": "assistant", "content": response_html}

        yield history, bars_html, gr.update()  # leave heatmap unchanged during streaming

    # Generation complete → render heatmap (delta scores)
    heatmap = render_heatmap(token_records)
    yield history, bars_html, gr.update(value=heatmap)


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

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Type a message and press Enter or click Send…",
                label="Message",
                scale=4,
                lines=1,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            with gr.Column(scale=3):
                # Gradio 6.x: messages format only, sanitize_html=False to allow spans
                # Fallback: if color spans are still stripped, replace this block with:
                #   chat_display = gr.HTML(elem_id="chat_display", label="Response")
                # and build full HTML manually in respond() instead of using history.
                chatbot = gr.Chatbot(
                    label="Response",
                    height=480,
                    render_markdown=True,
                    sanitize_html=False,    # allow <span style=...> for section coloring
                )

            with gr.Column(scale=2):
                bars_display = gr.HTML(
                    label="Emotion Scores",
                    value=render_emotion_bars({}),
                )

        heatmap_display = gr.Plot(label="Token-level Emotion Heatmap")

        # ---- Wire up events ---- #
        # Note: do NOT clear heatmap on msg_box.change — the .then() that clears
        # msg_box after submission would trigger it and erase the just-rendered heatmap.
        send_kwargs = dict(
            fn=respond,
            inputs=[msg_box, chatbot, mode_radio],
            outputs=[chatbot, bars_display, heatmap_display],
        )

        msg_box.submit(**send_kwargs).then(lambda: "", outputs=[msg_box])
        send_btn.click(**send_kwargs).then(lambda: "", outputs=[msg_box])

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

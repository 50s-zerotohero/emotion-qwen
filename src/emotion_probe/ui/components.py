"""UI components: emotion bar graph and token-level heatmap."""

import html
import numpy as np
import plotly.graph_objects as go

from emotion_probe.backend.base import TokenWithEmotions

# Section colors (used in both chatbot tokens and heatmap annotation)
SECTION_STYLES = {
    "think":      "background:#ede0f7; color:#5a2d82; border-radius:2px; padding:0 1px;",
    "scratchpad": "background:#fff3e0; color:#b45309; border-radius:2px; padding:0 1px;",
    "response":   "",
}

EMOTIONS = [
    "desperate", "calm", "sad", "happy", "nervous", "angry",
    "afraid", "guilty", "surprised", "loving", "inspired", "proud",
]

# Bar colors: positive → steel blue, negative → salmon
_BAR_POS = "#4472c4"
_BAR_NEG = "#c0392b"
_BAR_BG  = "#e8e8e8"


def color_token(token: str, section: str) -> str:
    """Wrap a token in a colored <span> based on its section."""
    style = SECTION_STYLES.get(section, "")
    escaped = html.escape(token).replace("\n", "<br>")
    if style:
        return f'<span style="{style}">{escaped}</span>'
    return escaped


def render_emotion_bars(emotions: dict[str, float]) -> str:
    """Render 12 emotion scores as an HTML bar chart.

    Bars are scaled relative to the max absolute value across all emotions
    for the current token, so relative magnitudes are visible.
    """
    if not emotions:
        return "<p style='color:#888'>Waiting for first token...</p>"

    scores = [emotions.get(e, 0.0) for e in EMOTIONS]
    max_abs = max(abs(s) for s in scores) or 1.0

    rows = []
    for emotion, score in zip(EMOTIONS, scores):
        pct = abs(score) / max_abs * 100
        bar_color = _BAR_POS if score >= 0 else _BAR_NEG
        sign = "+" if score >= 0 else ""

        # Label column (fixed width) + bar + value
        row = f"""
        <div style="display:flex; align-items:center; margin:2px 0; font-size:13px; font-family:monospace;">
          <span style="width:80px; text-align:right; padding-right:8px; color:#333;">{emotion}</span>
          <div style="flex:1; background:{_BAR_BG}; border-radius:3px; height:14px; position:relative;">
            <div style="width:{pct:.1f}%; background:{bar_color}; height:100%; border-radius:3px;"></div>
          </div>
          <span style="width:60px; text-align:right; padding-left:6px; color:{bar_color};">{sign}{score:.3f}</span>
        </div>"""
        rows.append(row)

    return (
        '<div style="padding:8px; border:1px solid #ddd; border-radius:6px; '
        'background:#fafafa; user-select:none;">'
        + "".join(rows)
        + "</div>"
    )


def render_heatmap(token_records: list[TokenWithEmotions]) -> go.Figure:
    """Render a token × emotion heatmap as a Plotly figure.

    Shown once after generation completes. Section boundaries are marked
    with vertical lines.
    """
    if not token_records:
        return go.Figure()

    n_tokens = len(token_records)
    z = np.zeros((len(EMOTIONS), n_tokens), dtype=np.float32)
    x_labels = []
    section_changes: list[int] = []   # token indices where section changes
    prev_section = token_records[0]["section"]

    for j, rec in enumerate(token_records):
        tok = rec["token"].replace("\n", "↵")[:8]   # truncate long tokens
        x_labels.append(f"{j}:{tok}")
        for i, emotion in enumerate(EMOTIONS):
            z[i, j] = rec["emotions"].get(emotion, 0.0)
        if rec["section"] != prev_section:
            section_changes.append(j)
            prev_section = rec["section"]

    abs_max = float(np.abs(z).max()) or 1.0

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=EMOTIONS,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-abs_max,
        zmax=abs_max,
        colorbar=dict(title="score", thickness=12),
    ))

    # Section boundary vertical lines
    for idx in section_changes:
        fig.add_vline(x=idx - 0.5, line_color="black", line_width=1.5, line_dash="dot")

    fig.update_layout(
        title="Token-level Emotion Activation",
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=11)),
        height=380,
        margin=dict(l=10, r=10, t=40, b=80),
    )
    return fig

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


def _bar_rows_html(emotions: dict[str, float], compact: bool = False) -> str:
    """Render inner bar rows. compact=True uses smaller font/height for dual-panel layout."""
    if not emotions:
        return "<p style='color:#888; margin:4px 0;'>Waiting for first token...</p>"

    scores = [emotions.get(e, 0.0) for e in EMOTIONS]
    max_abs = max(abs(s) for s in scores) or 1.0

    fs   = "11px" if compact else "13px"
    bh   = "9px"  if compact else "14px"
    mg   = "1px"  if compact else "2px"
    lw   = "68px" if compact else "80px"
    vw   = "52px" if compact else "60px"

    rows = []
    for emotion, score in zip(EMOTIONS, scores):
        pct = abs(score) / max_abs * 100
        bar_color = _BAR_POS if score >= 0 else _BAR_NEG
        sign = "+" if score >= 0 else ""
        row = (
            f'<div style="display:flex;align-items:center;margin:{mg} 0;'
            f'font-size:{fs};font-family:monospace;">'
            f'<span style="width:{lw};text-align:right;padding-right:6px;color:#333;">{emotion}</span>'
            f'<div style="flex:1;background:{_BAR_BG};border-radius:2px;height:{bh};position:relative;">'
            f'<div style="width:{pct:.1f}%;background:{bar_color};height:100%;border-radius:2px;"></div>'
            f'</div>'
            f'<span style="width:{vw};text-align:right;padding-left:5px;color:{bar_color};">'
            f'{sign}{score:.3f}</span>'
            f'</div>'
        )
        rows.append(row)
    return "".join(rows)


def _panel(title: str, body: str, compact: bool = False) -> str:
    pad = "5px 7px" if compact else "8px"
    return (
        f'<div style="padding:{pad};border:1px solid #ddd;border-radius:5px;'
        f'background:#fafafa;user-select:none;margin-bottom:6px;">'
        f'<div style="font-size:11px;font-weight:bold;color:#444;margin-bottom:3px;">{title}</div>'
        + body
        + "</div>"
    )


def render_emotion_bars(emotions: dict[str, float]) -> str:
    """Render 12 emotion scores as an HTML bar chart (single panel)."""
    return _panel("", _bar_rows_html(emotions))


def render_dual_emotion_bars(
    colon_scores: dict[str, float],
    live_scores: dict[str, float],
) -> str:
    """Render two compact stacked panels (fits within chatbot height of 480px)."""
    top    = _panel('Emotion at ":" token', _bar_rows_html(colon_scores, compact=True), compact=True)
    bottom = _panel("Live emotion (Δ from baseline)", _bar_rows_html(live_scores, compact=True), compact=True)
    return top + bottom


def render_heatmap(
    token_records: list[TokenWithEmotions],
    colon_scores: dict[str, float] | None = None,
) -> go.Figure:
    """Render a token × emotion heatmap as a Plotly figure.

    Column 0 is the ":" token (raw cosine scores at response start) when
    colon_scores is provided; subsequent columns show Δ from that baseline.
    Section boundaries are marked with vertical lines.
    """
    if not token_records:
        return go.Figure()

    # Build record list: replace the first entry (delta=0 all zeros) with
    # the raw colon_scores so the ":" column carries meaningful information.
    if colon_scores:
        colon_rec = dict(token_records[0])
        colon_rec["emotions"] = colon_scores
        all_records = [colon_rec] + list(token_records[1:])
    else:
        all_records = list(token_records)

    n_tokens = len(all_records)
    z = np.zeros((len(EMOTIONS), n_tokens), dtype=np.float32)
    x_labels = []
    section_changes: list[int] = []
    prev_section = all_records[0]["section"]

    for j, rec in enumerate(all_records):
        if j == 0 and colon_scores:
            x_labels.append(":")
        else:
            tok = rec["token"].replace("\n", "↵")[:8]
            x_labels.append(f"{j}:{tok}")
        for i, emotion in enumerate(EMOTIONS):
            z[i, j] = rec["emotions"].get(emotion, 0.0)
        if j > 0 and rec["section"] != prev_section:
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

    # Separator between ":" column and delta columns
    if colon_scores:
        fig.add_vline(x=0.5, line_dash="dash", line_color="black", line_width=2)

    # Section boundary vertical lines
    for idx in section_changes:
        fig.add_vline(x=idx - 0.5, line_color="black", line_width=1.5, line_dash="dot")

    fig.update_layout(
        title='Token-level Emotion Activation  |  col ":" = raw · rest = Δ from baseline',
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=11)),
        height=380,
        margin=dict(l=10, r=10, t=40, b=80),
    )
    return fig

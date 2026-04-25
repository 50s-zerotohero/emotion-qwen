"""
Script: 06_validate_emotion_probes.py

Generates validation figures:
  Fig 1 — 12×12 cosine similarity heatmap (hierarchically clustered)
  Fig 2 — PCA vs. human valence/arousal ratings (Russell & Mehrabian 1977)
  Fig 3 — Scenario × emotion cosine similarity heatmap (requires GPU)
  Fig 4A — 2D PCA scatter of emotion vectors
  Fig 4B — 3D interactive Plotly scatter (HTML)

Usage:
    python scripts/06_validate_emotion_probes.py          # all figures
    python scripts/06_validate_emotion_probes.py --no-gpu # skip Fig 3

Output:
    data/figures/fig1_cosine_similarity.png
    data/figures/fig2_pca_correlation.png
    data/figures/fig3_scenario_heatmap.png
    data/figures/fig4a_pca_2d.png
    data/figures/fig4b_pca_3d.html
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotion_probe.config import load_config

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "data" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Load emotion vectors
# --------------------------------------------------------------------------- #
_vec_path = PROJECT_ROOT / "data" / "emotion_vectors.pt"
_raw = torch.load(_vec_path, weights_only=True)
EMOTIONS = ["desperate", "calm", "sad", "happy", "nervous", "angry",
            "afraid", "guilty", "surprised", "loving", "inspired", "proud"]
V = np.stack([_raw[e].float().numpy() for e in EMOTIONS])  # (12, 2560)

# Unit-normalize
Vn = V / np.linalg.norm(V, axis=1, keepdims=True)

# --------------------------------------------------------------------------- #
# Human ratings — Russell & Mehrabian 1977 (valence, arousal)
# --------------------------------------------------------------------------- #
HUMAN_RATINGS = {
    "happy":     ( 0.81,  0.51),
    "inspired":  ( 0.62,  0.40),
    "loving":    ( 0.73,  0.26),
    "proud":     ( 0.63,  0.28),
    "calm":      ( 0.40, -0.48),
    "surprised": ( 0.13,  0.67),
    "guilty":    (-0.57,  0.10),
    "sad":       (-0.63, -0.27),
    "nervous":   (-0.49,  0.58),
    "afraid":    (-0.64,  0.60),
    "desperate": (-0.72,  0.38),
    "angry":     (-0.51,  0.59),
}

# --------------------------------------------------------------------------- #
# Scenarios for Figure 3 (texts must be >~80 tokens of content)
# --------------------------------------------------------------------------- #
SCENARIOS = {
    "Daughter's first steps": (
        "She had been holding her breath without realizing it. The baby stood beside the "
        "coffee table, fingers releasing the edge, and for one suspended moment simply stood "
        "there—alone, unsupported. Then one foot lifted and came forward, and the mother's "
        "hand flew to her mouth. The second step wobbled but held. Her daughter grinned at "
        "her with four teeth and the absolute confidence of someone who has just discovered "
        "they are capable of flight. She crossed the room in three steps and scooped the "
        "child up, pressing her face into the warm neck, laughing and crying at once."
    ),
    "Dog passed away": (
        "The house felt wrong in a way she couldn't name until she realized it was the "
        "silence. No clicking of nails on hardwood. No sigh from the corner of the room. "
        "She kept reaching down to scratch behind ears that were no longer there. His bowl "
        "still sat by the back door, and she couldn't bring herself to move it. That evening "
        "she sat on the kitchen floor, her back against the cabinet, and held the collar in "
        "both hands. Fourteen years. He had known her before her divorce, before the move, "
        "through everything. She stayed on the floor a long time."
    ),
    "Job interview nerves": (
        "His palms were sweating through his portfolio. The receptionist had told him it "
        "would be another fifteen minutes, which meant fifteen more minutes of rehearsing "
        "answers in his head, second-guessing the examples he'd chosen, wondering if his "
        "tie was too formal or not formal enough. He went over the gap in his resume again, "
        "practiced explaining it without sounding defensive. Every time the elevator opened "
        "he startled. He caught himself tapping his foot, made himself stop. The words "
        "'tell me about yourself' kept repeating in his mind, shapeless and enormous."
    ),
    "Eviction notice": (
        "The letter had been sitting on the counter for three days before she let herself "
        "read it properly. Thirty days. Her name in official type above a column of numbers "
        "she had been avoiding since November. She sat at the kitchen table and did the math "
        "again, and again the math didn't change. The school was in this district. Her mother "
        "was two blocks away. She had lived here for nine years. She opened her laptop and "
        "stared at the rental listings for a long time before closing it again, no closer "
        "to an answer than when she started."
    ),
    "PhD dissertation defense": (
        "The committee filed back in and he gripped the table edge. The chair set her notes "
        "face-down, looked up, and smiled before saying a word. Something unknotted in his "
        "chest. She said the words congratulations and Doctor and he heard them from somewhere "
        "slightly outside himself. His advisor shook his hand with both of hers. He had spent "
        "six years on this, had rewritten the third chapter four times, had sat in this same "
        "building at two in the morning wondering if it would ever be enough. Walking outside "
        "afterward, the afternoon light felt different, thicker, like something he'd earned."
    ),
    "Near-miss car accident": (
        "The other car came through the red light without slowing, and she had just enough "
        "time to yank the wheel before the impact she was already bracing for didn't happen. "
        "She pulled over half a block later, hands shaking so hard she couldn't get the car "
        "into park on the first try. In the rearview mirror she could see the other driver "
        "had continued on as if nothing had happened. Her heart was going so fast it felt "
        "arrhythmic. She sat with both hands pressed flat on her thighs and told herself "
        "to breathe, that it hadn't happened, that she was fine, though her body clearly "
        "hadn't received the update yet."
    ),
    "Betrayal discovered": (
        "She found the messages while he was in the shower. She hadn't meant to look—his "
        "phone had lit up with a name she didn't recognize and she'd glanced over by reflex. "
        "Then she read the thread. She set the phone back exactly as she had found it, went "
        "to the kitchen, and stood at the sink with the tap running cold over her wrists. "
        "She could hear the water in the pipes above her. She tried to organize what she "
        "knew into something that could be explained differently and could not. When he came "
        "downstairs she was looking out the window and did not turn around."
    ),
    "Unexpected lottery win": (
        "He checked the numbers twice, then a third time, then held the ticket under the "
        "lamp as if better light would change what was printed on it. He sat down on the "
        "couch, stood up, sat down again. The amount on the screen didn't look like a real "
        "number—it was the kind of figure he associated with other people's lives, with news "
        "stories, with things that happened elsewhere. He called his sister and when she "
        "answered he opened his mouth and discovered he had no idea what to say first. "
        "She asked if he was okay and he said he thought he might have won something."
    ),
    "Child critically ill": (
        "The monitor beside the bed made a sound every few seconds that she had learned "
        "to stop hearing. She hadn't left the hospital in two days. The chair had left "
        "marks on the backs of her legs. She watched her son's chest rise and fall, rise "
        "and fall, counted it the way she used to count his breaths when he was an infant "
        "and she couldn't sleep. The doctor had used the word waiting. She had nodded and "
        "understood that waiting was the only thing available to her and it was not enough "
        "and it was all there was."
    ),
    "Forgotten anniversary": (
        "She hadn't mentioned it all day and he hadn't noticed until after dinner, when "
        "she went to bed early and said she was tired. He stood in the kitchen and looked "
        "at the calendar on his phone and felt the specific horror of understanding something "
        "too late. He had been at work, then at the hardware store, then watching the game. "
        "He had not thought of it once. He went upstairs and she was not asleep. He said "
        "her name and she said it was fine, which was the clearest possible indication that "
        "it was not fine. He sat on the edge of the bed and tried to find words that were "
        "not excuses."
    ),
    "Morning meditation": (
        "The lake was completely still. She had arrived before the birds, before the runners, "
        "before any other sound. She sat on the dock with her legs crossed and let the cold "
        "air settle around her. Her thoughts moved across the surface of her mind and "
        "continued past without snagging. The light came up slowly and turned the water "
        "from grey to silver to a pale dusty blue. A heron appeared at the far end and stood "
        "motionless for a long time. She breathed in for four counts and out for four counts "
        "and felt the space between moments expand until time became something that simply "
        "wasn't relevant."
    ),
    "Standing ovation": (
        "She had prepared for the possibility of polite applause. What happened instead was "
        "that the first row stood, and then the second, and then it moved back through the "
        "auditorium like a wave she could see coming before it reached her. She stood at the "
        "podium with her notes still in her hand and did not know where to look. She had "
        "worked on this research for four years. She had given the talk to her bathroom mirror "
        "dozens of times. Nothing had prepared her for the sound of six hundred people "
        "applauding as if they meant it. Her face hurt from smiling before she realized "
        "she was smiling."
    ),
}

# Color palette consistent across figures
EMOTION_COLORS = {
    "happy": "#f4c430", "inspired": "#ff8c00", "loving": "#e83e8c",
    "proud": "#8b4513", "calm": "#4a9eca", "surprised": "#9b59b6",
    "guilty": "#95a5a6", "sad": "#3498db", "nervous": "#f39c12",
    "afraid": "#e74c3c", "desperate": "#c0392b", "angry": "#922b21",
}


# --------------------------------------------------------------------------- #
# Figure 1 — 12×12 cosine similarity heatmap (hierarchically clustered)
# --------------------------------------------------------------------------- #
def figure1_cosine_heatmap():
    print("Generating Figure 1: cosine similarity heatmap...")
    cos = Vn @ Vn.T  # (12, 12)

    # Hierarchical clustering: reorder emotions
    dist = 1.0 - np.clip(cos, -1, 1)
    np.fill_diagonal(dist, 0.0)
    linked = linkage(dist[np.triu_indices(12, k=1)], method="average")
    order = leaves_list(linked)

    emo_ord = [EMOTIONS[i] for i in order]
    cos_ord = cos[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cos_ord, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")

    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels(emo_ord, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(emo_ord, fontsize=10)

    # Annotate cells
    for i in range(12):
        for j in range(12):
            val = cos_ord[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title("Emotion Vector Cosine Similarity\n(hierarchically clustered)",
                 fontsize=13, pad=12)
    fig.tight_layout()
    out = FIGURES_DIR / "fig1_cosine_similarity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# --------------------------------------------------------------------------- #
# Figure 2 — PCA vs. human valence/arousal ratings
# --------------------------------------------------------------------------- #
def figure2_pca_correlation():
    print("Generating Figure 2: PCA vs. human ratings...")
    pca = PCA(n_components=5)
    projected = pca.fit_transform(V)  # (12, 5)

    valence = np.array([HUMAN_RATINGS[e][0] for e in EMOTIONS])
    arousal = np.array([HUMAN_RATINGS[e][1] for e in EMOTIONS])

    # Report PC1–PC5 correlations with both axes
    print("  PC vs valence/arousal correlations:")
    for pc_idx in range(5):
        r_val, p_val = pearsonr(projected[:, pc_idx], valence)
        r_aro, p_aro = pearsonr(projected[:, pc_idx], arousal)
        var = pca.explained_variance_ratio_[pc_idx] * 100
        print(f"  PC{pc_idx+1} ({var:.1f}% var): "
              f"r_valence={r_val:+.3f} (p={p_val:.3f})  "
              f"r_arousal={r_aro:+.3f} (p={p_aro:.3f})")

    # Select best PC for each axis by highest |r|
    r_vals = [pearsonr(projected[:, i], valence)[0] for i in range(5)]
    r_aros = [pearsonr(projected[:, i], arousal)[0] for i in range(5)]
    best_val_idx = int(np.argmax(np.abs(r_vals)))
    best_aro_idx = int(np.argmax(np.abs(r_aros)))
    print(f"  → Best for valence: PC{best_val_idx+1} (r={r_vals[best_val_idx]:+.3f})")
    print(f"  → Best for arousal: PC{best_aro_idx+1} (r={r_aros[best_aro_idx]:+.3f})")

    # Build plot axes: PC1 vs valence, best arousal PC vs arousal
    pc_val = projected[:, best_val_idx].copy()
    pc_aro = projected[:, best_aro_idx].copy()

    # Flip sign so positive PC = positive human rating
    if r_vals[best_val_idx] < 0:
        pc_val = -pc_val
    if r_aros[best_aro_idx] < 0:
        pc_aro = -pc_aro

    r1, p1 = pearsonr(pc_val, valence)
    r2, p2 = pearsonr(pc_aro, arousal)
    var_val = pca.explained_variance_ratio_[best_val_idx] * 100
    var_aro = pca.explained_variance_ratio_[best_aro_idx] * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (x, y, xlabel, ylabel, r, p) in zip(axes, [
        (pc_val, valence,
         f"PC{best_val_idx+1} ({var_val:.1f}% var)", "Valence (human)", r1, p1),
        (pc_aro, arousal,
         f"PC{best_aro_idx+1} ({var_aro:.1f}% var)", "Arousal (human)", r2, p2),
    ]):
        for i, emo in enumerate(EMOTIONS):
            ax.scatter(x[i], y[i], color=EMOTION_COLORS[emo], s=80, zorder=3)
            ax.annotate(emo, (x[i], y[i]), textcoords="offset points",
                        xytext=(6, 4), fontsize=9)

        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, "k--", alpha=0.5, lw=1)

        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)

        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.set_title(f"r = {r:.3f}  {p_str}", fontsize=12)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Emotion Vectors: Best PC vs. Human Valence/Arousal Ratings\n"
                 "(Russell & Mehrabian 1977)", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_pca_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# --------------------------------------------------------------------------- #
# Figure 3 — Scenario × emotion cosine similarity heatmap (GPU)
# --------------------------------------------------------------------------- #
def figure3_scenario_heatmap(cfg: dict):
    print("Generating Figure 3: scenario × emotion heatmap (GPU)...")
    from emotion_probe.probe.activation_recorder import record_activations

    scenario_names = list(SCENARIOS.keys())
    scenario_texts = list(SCENARIOS.values())

    # Get mean activations for each scenario (shape: n_scenarios × 2560)
    acts = record_activations(scenario_texts, cfg)

    # Check for nan rows (texts too short)
    nan_mask = np.isnan(acts).any(axis=1)
    if nan_mask.any():
        for i, name in enumerate(scenario_names):
            if nan_mask[i]:
                print(f"  WARNING: '{name}' was skipped (text too short)")

    # Cosine similarity: scenario activations vs unit emotion vectors
    acts_t = torch.tensor(acts).float()
    # Normalize each row, handling nan rows
    norms = acts_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    acts_n = acts_t / norms
    acts_n[nan_mask] = float("nan")

    Vn_t = torch.tensor(Vn).float()
    sim = (acts_n @ Vn_t.T).numpy()  # (n_scenarios, 12)

    # Row-wise 99th-percentile normalization (paper-compliant)
    sim_norm = sim.copy()
    for i in range(sim_norm.shape[0]):
        row = sim_norm[i, :]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            p99 = np.percentile(np.abs(valid), 99)
            if p99 > 0:
                sim_norm[i, :] = row / p99

    fig, ax = plt.subplots(figsize=(10, 7))
    vmax = np.nanmax(np.abs(sim_norm))
    im = ax.imshow(sim_norm, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
                 label="Cosine similarity (99th-pct normalized per scenario)")

    ax.set_xticks(range(12))
    ax.set_yticks(range(len(scenario_names)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(scenario_names, fontsize=10)

    for i in range(len(scenario_names)):
        for j in range(12):
            val = sim_norm[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=color)

    ax.set_title("Scenario × Emotion Cosine Similarity\n"
                 "(Qwen3-4B layer 20, ChatML-aligned, row 99th-pct normalized)",
                 fontsize=13, pad=12)
    fig.tight_layout()
    out = FIGURES_DIR / "fig3_scenario_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# --------------------------------------------------------------------------- #
# Figure 4A — 2D PCA scatter of emotion vectors
# --------------------------------------------------------------------------- #
_POSITIVE_EMOTIONS = {"happy", "loving", "proud", "inspired", "calm"}
_NEGATIVE_EMOTIONS = {"desperate", "sad", "afraid", "nervous", "angry", "guilty"}


def _valence_color(emo: str) -> str:
    if emo in _POSITIVE_EMOTIONS:
        return "#c0392b"   # red family
    if emo in _NEGATIVE_EMOTIONS:
        return "#2980b9"   # blue family
    return "#7f8c8d"       # gray (surprised)


def figure4a_pca_2d():
    print("Generating Figure 4A: 2D PCA scatter of emotion vectors...")
    pca2 = PCA(n_components=2)
    proj2 = pca2.fit_transform(Vn)  # (12, 2)

    var1 = pca2.explained_variance_ratio_[0] * 100
    var2 = pca2.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(9, 8))

    for i, emo in enumerate(EMOTIONS):
        c = _valence_color(emo)
        ax.scatter(proj2[i, 0], proj2[i, 1], color=c, s=120, zorder=3)
        ax.annotate(emo, (proj2[i, 0], proj2[i, 1]),
                    textcoords="offset points", xytext=(7, 4), fontsize=10)

    # Grand mean is mean of data = PCA center → projects to (0, 0)
    ax.scatter(0, 0, marker="x", color="black", s=120, linewidths=2,
               zorder=4, label="grand mean")

    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.axvline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel(f"PC1 ({var1:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var2:.1f}% var)", fontsize=12)
    ax.set_title("Emotion Vectors — 2D PCA\n"
                 "Red = positive valence · Blue = negative valence",
                 fontsize=13, pad=12)

    # Legend patches
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#c0392b", label="positive valence"),
        mpatches.Patch(color="#2980b9", label="negative valence"),
        mpatches.Patch(color="#7f8c8d", label="neutral (surprised)"),
        plt.Line2D([0], [0], marker="x", color="black", lw=0, markersize=10,
                   markeredgewidth=2, label="grand mean"),
    ], fontsize=9, loc="best")

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = FIGURES_DIR / "fig4a_pca_2d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# --------------------------------------------------------------------------- #
# Figure 4B — 3D interactive Plotly scatter (HTML)
# --------------------------------------------------------------------------- #
def figure4b_pca_3d():
    print("Generating Figure 4B: 3D Plotly scatter of emotion vectors...")
    import plotly.graph_objects as go

    pca3 = PCA(n_components=3)
    proj3 = pca3.fit_transform(Vn)  # (12, 3)

    var1 = pca3.explained_variance_ratio_[0] * 100
    var2 = pca3.explained_variance_ratio_[1] * 100
    var3 = pca3.explained_variance_ratio_[2] * 100

    colors  = [_valence_color(e) for e in EMOTIONS]

    fig = go.Figure()

    # Emotion points
    fig.add_trace(go.Scatter3d(
        x=proj3[:, 0], y=proj3[:, 1], z=proj3[:, 2],
        mode="markers+text",
        text=EMOTIONS,
        textposition="top center",
        marker=dict(size=8, color=colors, opacity=0.85),
        name="emotions",
    ))

    # Grand mean at PCA center (0, 0, 0)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=10, color="black", symbol="cross"),
        name="grand mean",
    ))

    fig.update_layout(
        title="Emotion Vectors — 3D PCA (rotate & zoom)",
        scene=dict(
            xaxis_title=f"PC1 ({var1:.1f}%)",
            yaxis_title=f"PC2 ({var2:.1f}%)",
            zaxis_title=f"PC3 ({var3:.1f}%)",
        ),
        width=900, height=750,
        legend=dict(x=0.02, y=0.98),
    )

    out = FIGURES_DIR / "fig4b_pca_3d.html"
    fig.write_html(str(out))
    print(f"  Saved → {out}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gpu", action="store_true",
                        help="Skip Figure 3 (requires GPU)")
    args = parser.parse_args()

    figure1_cosine_heatmap()
    figure2_pca_correlation()

    if not args.no_gpu:
        cfg = load_config()
        figure3_scenario_heatmap(cfg)
    else:
        print("Skipping Figure 3 (--no-gpu flag set)")

    figure4a_pca_2d()
    figure4b_pca_3d()

    print("\nAll figures saved to data/figures/")


if __name__ == "__main__":
    main()

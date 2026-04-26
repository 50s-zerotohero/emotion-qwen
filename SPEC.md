# Emotion Probe for Qwen3 — Project Specification (v0.2.0)

## Project Overview

A research reproduction of Anthropic's 2026 paper
"Emotion Concepts and their Function in a Large Language Model"
using **Qwen3-4B Dense** as the target model.

**Goals:**
1. Extract linear emotion direction vectors from Qwen3-4B's residual stream
2. Validate them against the paper's PCA / correlation benchmarks
3. Build a real-time Gradio UI that visualizes emotion activations token-by-token during inference
4. Support activation steering (adding emotion vectors to hidden states mid-generation)

**References:**
- Paper: https://transformer-circuits.pub/2026/emotions/index.html
- Interactive tool: https://transformer-circuits.pub/2026/emotions/onpolicy/index.html

---

## Environment

| Item | Value |
|------|-------|
| Python | 3.12 |
| PyTorch | nightly cu128 |
| GPU | RTX 5090 (32 GB VRAM) |
| Model VRAM | ~10 GB (bfloat16) |
| nnsight | 0.6.x |
| Gradio | 6.x |

---

## 12 Emotions

| Emotion | Valence | Arousal |
|---------|---------|---------|
| desperate | low | high |
| calm | mid-high | low |
| sad | low | low |
| happy | high | mid |
| nervous | low | high |
| angry | low | high |
| afraid | low | high |
| guilty | low | mid |
| surprised | mid | high |
| loving | high | low-mid |
| inspired | high | high |
| proud | high | mid |

---

## Model Configuration (`config.yaml`)

```yaml
model:
  name: "Qwen/Qwen3-4B"
  dtype: "bfloat16"
  device: "cuda"

emotions:
  - desperate
  - calm
  - sad
  - happy
  - nervous
  - angry
  - afraid
  - guilty
  - surprised
  - loving
  - inspired
  - proud

story_generation:
  stories_per_emotion: 50
  min_words: 80
  max_words: 150
  min_tokens: 80
  max_tokens: 250

neutral_generation:
  n_neutral_texts: 200
  min_words: 80
  max_words: 150

extraction:
  skip_first_n_tokens: 50   # paper-compliant; skip header + early tokens
  layer: 20                 # layer 20/36 — optimal for linear probe

noise_removal:
  enabled: true
  variance_explained: 0.50  # remove top neutral PCs up to 50% cumulative variance

reasoning:
  max_new_tokens: 512
  temperature: 0.7
```

---

## Directory Structure

```
emotion-qwen/
├── SPEC.md                              # this file
├── CLAUDE.md                            # Claude Code instructions
├── config.yaml
├── .env.example                         # ANTHROPIC_API_KEY, HF_TOKEN
├── .gitignore                           # includes .env, *.pt
├── pyproject.toml
│
├── src/emotion_probe/
│   ├── config.py                        # load_config(), HF_TOKEN
│   ├── backend/
│   │   ├── base.py                      # EmotionProbeBackend ABC, TokenWithEmotions
│   │   └── local_nnsight.py             # LocalNNSightBackend (main implementation)
│   ├── probe/
│   │   ├── story_generation.py          # Claude API story generation
│   │   ├── neutral_generation.py        # neutral text generation
│   │   ├── activation_recorder.py       # nnsight residual stream capture
│   │   ├── emotion_vectors.py           # mean-diff + noise removal → final vectors
│   │   └── noise_removal.py             # PCA noise PC projection
│   └── ui/
│       ├── app.py                       # Gradio app (build_app, launch)
│       └── components.py                # HTML bars, heatmap, dual panel
│
├── scripts/
│   ├── 01_generate_stories.py           # generate 600 stories + 200 neutral texts
│   ├── 02_verify_story_lengths.py       # token-length distribution check
│   ├── 03_extract_vectors.py            # extract emotion vectors
│   ├── 04_verify_vectors.py             # cosine heatmap sanity check
│   ├── 05_launch_ui.py                  # launch Gradio UI
│   ├── 06_validate_emotion_probes.py    # PCA, correlation, scenario heatmap figures
│   ├── 07_steering_poem_experiment.py   # steering → poem generation + Claude eval
│   └── 08_logit_lens.py                 # emotion vectors → vocabulary space
│
└── data/
    ├── stories/
    │   ├── emotion_stories.json         # 12 emotions × 50 = 600 stories
    │   └── neutral_texts.json           # 200 neutral texts
    ├── activations/
    │   ├── neutral_pca_basis.pt         # noise removal PCA basis
    │   └── raw_emotion_means.pt         # pre-projection means (debug)
    ├── emotion_vectors.pt               # final emotion direction vectors
    ├── steering_poems/                  # output of script 07
    └── figures/                         # output of script 06
```

---

## Core Abstractions

### `TokenWithEmotions` (base.py)

```python
class TokenWithEmotions(TypedDict):
    token: str
    section: str        # "think" | "scratchpad" | "response"
    emotions: dict[str, float]
```

### `EmotionProbeBackend` (base.py)

```python
class EmotionProbeBackend(ABC):
    @abstractmethod
    async def generate_with_emotions(
        self,
        user_message: str,
        mode: ReasoningMode,
        emotion_vectors: dict[str, torch.Tensor],
        max_new_tokens: int = 512,
        system_prompt: str = "",
        steering_emotion: str | None = None,
        steering_alpha: float = 0.0,
    ) -> AsyncIterator[TokenWithEmotions]: ...
```

### `ReasoningMode` (base.py)

```python
class ReasoningMode(str, Enum):
    NO_THINK   = "no_think"    # inject <think></think> prefix → skip thinking
    THINK      = "think"       # standard Qwen3 template → model generates <think>
    SCRATCHPAD = "scratchpad"  # NO_THINK + <SCRATCHPAD_REASONING> prefix
```

---

## Completed Pipeline (Steps 1–8)

### Step 1 — Scaffold
Set up `pyproject.toml`, `config.yaml`, `.env.example`, `.gitignore`, package structure.

### Step 2 — Story Generation
- `story_generation.py`: Claude API generates 50 stories per emotion (600 total)
- `neutral_generation.py`: 200 neutral factual texts (encyclopedia-style)
- Stories: 80–150 words, third-person, avoid the emotion word itself

### Step 3 — Activation Extraction

**ChatML wrapping (critical):** Stories are wrapped as a ChatML assistant reply
using `wrap_as_assistant()` before extraction. This ensures the model's emotion
representations are sampled from the same distribution as actual inference —
plain-text stories lie in a different part of the representation space.

```
<|im_start|>assistant\n{story}\n<|im_end|>
```

Extraction procedure:
1. Forward pass each story through `lm.model.layers[layer_idx]` (layer 20)
2. Capture `output[0]` — shape `(seq_len, hidden_dim)` in bfloat16
3. Skip header tokens + first 50 content tokens
4. Mean-pool remaining positions → per-story mean activation
5. Compute per-emotion mean over 50 stories
6. Subtract grand mean (mean over all 12 emotions) → mean-diff vectors
7. Project out top-1 noise PC from neutral texts (removes formatting bias)
8. Save to `data/emotion_vectors.pt`

### Step 4 — Verification
Cosine similarity heatmap confirms diagonal dominance (each emotion most similar
to itself). Confirmed `desperate` bias was resolved after ChatML wrapping fix.

### Step 5 — Gradio UI
`scripts/05_launch_ui.py` → launches the real-time visualization app.

### Step 6 — Validation (`scripts/06_validate_emotion_probes.py`)

Produces three figures:

**Figure 2: PCA scatter (2D)**
- 12 emotion vectors projected to PC1 × PC2
- PC1 (30.0% variance): r = −0.831 with human valence ratings (p < 0.001)
- PC4 (9.1% variance): r = +0.707 with human arousal ratings (p = 0.010)
- Color: positive valence = red (#c0392b), negative = blue (#2980b9), neutral = gray

**Figure 2b: PCA scatter (3D, interactive)**
- PC1 × PC2 × PC4; saved as `data/figures/fig4b_pca_3d.html`

**Figure 3: Scenario heatmap**
- 12 emotions × N scenarios cosine similarity matrix
- Row-wise 99th-percentile normalization (paper-compliant): `row / p99(|row|)`
- Reveals clear emotional signatures per scenario

### Step 7 — Steering Poem Experiment (`scripts/07_steering_poem_experiment.py`)
- Prompt: "Please create a poem within 50 words."
- Generate baseline + 12 steered poems (α = 5.0)
- Evaluate each poem with Claude API (emotional_tones, imagery, valence)
- Results: 1/12 (8.3%) exact name match; ~7–8/12 (58–67%) semantic match

### Step 8 — Logit Lens (`scripts/08_logit_lens.py`)
- Project each emotion vector through `lm_head.weight` (unembedding matrix)
- `logits = W @ vec_unit` where `W.shape = (vocab_size, hidden_dim)`
- Shows top/bottom 10 tokens per emotion
- Finding: cross-lingual tokens (Chinese + English) in Qwen3 vocabulary

---

## nnsight Usage Notes (Qwen3-4B, v0.6.x)

```python
# Correct layer access pattern
with lm.trace(input_ids, remote=False):
    hidden = lm.model.layers[layer_idx].output[0].save()
    # shape: (seq_len, hidden_dim) — NO batch dim
    logits = lm.lm_head.output.save()
    # shape: (1, seq_len, vocab_size) — HAS batch dim

# .save() returns torch.Tensor directly (not a proxy with .value)
hidden_tensor = hidden  # already a tensor
```

**lm_head.weight is a meta tensor outside a trace.**
To materialize it, run a minimal forward pass:
```python
with lm.trace(input_ids, remote=False):
    w_save = lm.lm_head.weight.save()
W = (w_save if isinstance(w_save, torch.Tensor) else w_save.value).detach().float().cpu()
```

---

## Activation Steering

Implemented in `LocalNNSightBackend.generate_with_emotions()`:

```python
# Inside lm.trace():
_h_norm   = lm.model.layers[layer_idx].output[0][-1, :].norm()
_effective = steering_alpha * _h_norm / 10.0
lm.model.layers[layer_idx].output[0][:] = (
    lm.model.layers[layer_idx].output[0] + _effective * sv_gpu
)
logit_save = lm.lm_head.output.save()
```

**Norm-proportional scaling:** `eff = α × hidden_norm / 10`
- Rationale: Qwen3-4B hidden norm ≈ 60 at layer 20.
  A unit vector is nearly orthogonal to the hidden state, so a fixed-magnitude
  addition has negligible effect. Scaling by `norm/10` makes α=10 produce a
  perturbation equal to the hidden-state norm — clearly audible in generation.
- Recommended α range: ±1–5 for subtle effects, ±8–10 for strong effects.

**Note:** `hidden_save` is computed before the in-place modification because
nnsight's in-place update also affects the already-saved tensor. Emotion scoring
uses the pre-steering hidden state to avoid circularity.

---

## Gradio UI Features (`src/emotion_probe/ui/`)

### Layout
```
[Reasoning Mode radio]
[System Prompt textbox]
[Steering emotion dropdown | Steering α slider]
[Message box | Send | Stop]
[Chatbot (left) | Dual emotion panel (right)]
[Token-level heatmap (full width, shown after generation)]
```

### Reasoning Mode
- `no_think`: injects `<think>\n\n</think>` prefix → model skips thinking
- `think`: standard Qwen3 generation with `<think>…</think>` block
- `scratchpad`: no_think + `<SCRATCHPAD_REASONING>` prefix for custom scratchpad

### Section Coloring
Tokens colored by section in the chatbot:
- think section: purple (#5a2d82)
- scratchpad section: amber (#b45309)
- response section: default

### Dual Emotion Panel (`render_dual_emotion_bars`)
- **Top panel:** "Emotion at ':' token" — raw cosine scores at first generated token (fixed)
- **Bottom panel:** "Live emotion (Δ from baseline)" — current token score minus baseline

### Heatmap (`render_heatmap`)
- X-axis: generated tokens; Y-axis: 12 emotions
- First column: ":" (raw scores at generation start); remaining: Δ from baseline
- Black dashed vertical line separates ":" column from Δ columns

### Stop Button
Wired with `cancels=[submit_gen, click_gen]` targeting the generator events directly:
```python
submit_gen = msg_box.submit(**send_kwargs)
submit_gen.then(lambda: "", outputs=[msg_box])
click_gen = send_btn.click(**send_kwargs)
click_gen.then(lambda: "", outputs=[msg_box])
stop_btn.click(fn=None, cancels=[submit_gen, click_gen])
```

---

## Key Discoveries

### 1. ChatML Distribution Gap
Extracting activations from plain-text stories misses the actual assistant-mode
distribution. Wrapping stories as `<|im_start|>assistant\n…\n<|im_end|>` aligns
the extraction context with inference context — resolving the `desperate` column
bias seen in early experiments.

### 2. PC4 Captures Arousal
The paper reports PC1 ≈ valence. In this reproduction:
- **PC1** (30.0%): r = −0.831 with valence (p < 0.001) ✓
- **PC4** (9.1%): r = +0.707 with arousal (p = 0.010)

Arousal is encoded orthogonally to valence, consistent with the circumplex model
of emotion.

### 3. Cross-Lingual Emotion Tokens (Logit Lens)
Projecting emotion vectors through the unembedding matrix reveals that Qwen3's
emotion directions activate both Chinese and English vocabulary tokens —
consistent with multilingual pretraining.

### 4. RLHF Atmospheric Imagery Transformation
Steering with emotion vectors in poems does not produce raw emotional words.
Instead, the model routes through RLHF-trained imagery (atmospheric descriptions,
sensory metaphors) — reflecting how RLHF fine-tuning has reshaped the surface
expression of internal emotional states.

---

## Validation Results Summary

| Metric | Value | Threshold |
|--------|-------|-----------|
| PC1 × valence correlation | r = −0.831 (p < 0.001) | r > 0.7 |
| PC4 × arousal correlation | r = +0.707 (p = 0.010) | r > 0.5 |
| Cosine heatmap diagonal mean | > 0.85 | > 0.7 |
| Logit lens semantic match | ~83% (10/12) | > 70% |
| Poem steering semantic match | ~58–67% (7–8/12) | > 50% |

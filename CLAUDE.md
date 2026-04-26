# CLAUDE.md

You are working on **Emotion Probe for Qwen3**, a research reproduction of
Anthropic's 2026 "Emotion Concepts" paper using Qwen3-4B Dense.

Read `SPEC.md` first to understand the full design. It is the canonical reference.

## Working Style

- Research code — prioritize readability and reproducibility over robustness
- Communicate in Japanese with the user (Japanese researcher)
- Code: English identifiers, English comments

## Key Files

| File | Role |
|------|------|
| `config.yaml` | Model name, layer, emotion list, generation params |
| `src/emotion_probe/backend/local_nnsight.py` | Core: nnsight generation loop, steering |
| `src/emotion_probe/ui/app.py` | Gradio app entry point |
| `src/emotion_probe/ui/components.py` | Emotion bars, heatmap rendering |
| `src/emotion_probe/probe/activation_recorder.py` | Residual stream capture |
| `data/emotion_vectors.pt` | Final emotion direction vectors (not committed) |

## Commit Rules

- Commit after each completed step
- Format: `[Step N] Short English description`
- Stage specific files — never `git add -A`
- Verify `.gitignore` covers `.env` and `*.pt` before every commit

## Files NEVER to Commit

- `.env` (API keys, HF token)
- `data/**/*.pt` (large binary tensors)
- `data/stories/*.json` (generated data, large)

## Scripts Convention

`scripts/NN_description.py` — numbered, run from project root:
```
python scripts/05_launch_ui.py
```
Each script adds `PROJECT_ROOT/src` to `sys.path` at the top.

## Python Path

All scripts prepend:
```python
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
```
This allows `from emotion_probe.xxx import yyy` without installing the package.

## GPU / Memory Notes

- RTX 5090 (32 GB VRAM); Qwen3-4B in bfloat16 uses ~10 GB
- nnsight 0.6.x: always use `remote=False` for local execution
- Layer access: `lm.model.layers[i].output[0]` → `(seq_len, hidden_dim)` (no batch dim)
- `lm.lm_head.weight` is a meta tensor outside a trace — must `.save()` inside trace

## API Keys

- Access only via `os.getenv()` / `load_dotenv()`
- Never print or log key values

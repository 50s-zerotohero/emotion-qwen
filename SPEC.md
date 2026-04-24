# Emotion Probe for Qwen3 — Project Specification (v2)

## プロジェクト目的

Anthropicの論文 "Emotion Concepts and their Function in a Large Language Model" (2026/04)
の感情ベクトル抽出パイプラインを、Qwen3-4B Denseで再現する。
さらに、論文のインタラクティブ可視化ツールと同等の、
**リアルタイムで応答中の感情活性化を可視化するチャットUI** をローカルで構築する。

参考:
- 論文: https://transformer-circuits.pub/2026/emotions/index.html
- インタラクティブツール: https://transformer-circuits.pub/2026/emotions/onpolicy/index.html
- ユーザーのZennブログ(論文解説): https://zenn.dev/50s_zerotohero/articles/bf2af2d8a6e608
- ハルシネーションプローブ実装参考: https://zenn.dev/50s_zerotohero/articles/2b85e9a3668a7b
- 先行SLM再現研究: https://arxiv.org/abs/2604.04064
- Gemma4-e4b再現例 (lyra bubbles, X 2026/04/13): 「a bit noisy but it works」

## 実験対象の12感情

論文のインタラクティブツールで使用されている感情群に揃える。

| 感情 | Valence (快-不快) | Arousal (覚醒度) |
|---|---|---|
| desperate | 低 | 高 |
| calm | 中〜高 | 低 |
| sad | 低 | 低 |
| happy | 高 | 中 |
| nervous | 低 | 高 |
| angry | 低 | 高 |
| afraid | 低 | 高 |
| guilty | 低 | 中 |
| surprised | 中 | 高 |
| loving | 高 | 低〜中 |
| inspired | 高 | 高 |
| proud | 高 | 中 |

## 技術スタック

| レイヤ | 技術 |
|---|---|
| モデル | Qwen3-4B (Dense) |
| 内部活性化取得 | nnsight |
| バックエンド抽象 | Python abstract base class |
| Phase 1 バックエンド | LocalNNSightBackend (RTX 5090) |
| Phase 2 バックエンド | 抽象インターフェースのみ(Modal想定) |
| ストーリー生成 | Claude API (claude-sonnet-4-6) |
| UI | Gradio (gr.Blocks) |
| 可視化 | Plotly (heatmap) + HTML/CSS bars |
| パッケージ管理 | uv or pip + requirements.txt |
| Python | 3.12 |

## アーキテクチャ

### ディレクトリ構造

```
emotion-qwen/
├── README.md
├── SPEC.md                          # この仕様書
├── CLAUDE.md                        # Claude Code向け指示
├── pyproject.toml
├── .env.example                     # ANTHROPIC_API_KEY, HF_TOKEN
├── .gitignore
├── config.yaml                      # モデル名、層番号、感情リスト
│
├── src/emotion_probe/
│   ├── __init__.py
│   ├── config.py                    # 設定ロード
│   │
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── base.py                  # EmotionProbeBackend 抽象基底
│   │   ├── local_nnsight.py         # Phase 1 本体
│   │   └── modal_vllm.py            # Phase 2 スタブ
│   │
│   ├── probe/
│   │   ├── __init__.py
│   │   ├── story_generation.py      # Claude APIで感情ストーリー生成
│   │   ├── neutral_generation.py    # Claude APIで中立テキスト生成
│   │   ├── activation_recorder.py   # nnsightで残差ストリーム記録
│   │   ├── emotion_vectors.py       # ベクトル抽出・保存・読込
│   │   └── noise_removal.py         # 中立テキストのtop PC計算・射影除去
│   │
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── modes.py                 # ReasoningMode Enum
│   │   └── prompts.py               # モード別プロンプトテンプレート
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py                   # Gradio メインエントリ
│   │   ├── components.py            # UI部品(BarGraph, Heatmap)
│   │   └── theme.py                 # カラーパレット
│   │
│   └── utils/
│       ├── __init__.py
│       └── logging.py
│
├── scripts/
│   ├── 01_generate_stories.py       # 感情ストーリー + 中立テキスト生成
│   ├── 02_verify_story_lengths.py   # トークン数分布の事前検証
│   ├── 03_extract_vectors.py        # 感情ベクトル抽出(+ノイズ除去)
│   ├── 04_verify_vectors.py         # 抽出結果の検証(可視化)
│   └── 05_launch_ui.py              # UI起動
│
├── data/
│   ├── stories/
│   │   ├── emotion_stories.json     # 12感情 × 50 = 600ストーリー
│   │   └── neutral_texts.json       # 中立テキスト 200本
│   ├── activations/                 # 中間生成物
│   │   ├── neutral_pca_basis.pt     # ノイズ除去用のPC基底
│   │   └── raw_emotion_means.pt     # 射影除去前のベクトル(デバッグ用)
│   └── emotion_vectors.pt           # 最終成果物(射影除去済み)
│
├── notebooks/
│   └── exploration.ipynb            # 探索用
│
└── tests/
    ├── test_story_generation.py
    ├── test_emotion_vectors.py
    ├── test_noise_removal.py
    └── test_backends.py
```

### config.yaml

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
  min_tokens: 80      # Qwen3 tokenizer で測定、これ未満は除外
  max_tokens: 250

neutral_generation:
  n_neutral_texts: 200
  min_words: 80
  max_words: 150

extraction:
  skip_first_n_tokens: 50   # 論文準拠、感情的内容が明確になる位置
  layer: 20                 # Qwen3-4B の抽出層 (36層中、後半寄り)

noise_removal:
  enabled: true             # Phase 1 デフォルトで有効
  variance_explained: 0.50  # 除去する主成分の累積寄与率

reasoning:
  max_new_tokens: 512
  temperature: 0.7
```

### 主要インターフェース

#### `src/emotion_probe/reasoning/modes.py`

```python
from enum import Enum

class ReasoningMode(str, Enum):
    NO_THINK = "no_think"      # <think></think> 空
    THINK = "think"            # <think>...</think> あり (Qwen3デフォルト)
    SCRATCHPAD = "scratchpad"  # <think></think> + <SCRATCHPAD_REASONING>...</SCRATCHPAD_REASONING>
```

#### `src/emotion_probe/backend/base.py`

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, TypedDict
import torch

class TokenWithEmotions(TypedDict):
    token: str
    section: str  # "think" | "scratchpad" | "response"
    emotions: dict[str, float]  # {"desperate": 0.23, ...}

class EmotionProbeBackend(ABC):
    @abstractmethod
    async def generate_with_emotions(
        self,
        user_message: str,
        mode: ReasoningMode,
        emotion_vectors: dict[str, torch.Tensor],
        max_new_tokens: int = 512,
    ) -> AsyncIterator[TokenWithEmotions]:
        """トークンごとに感情活性化を返すストリーミング生成"""
        pass

    @abstractmethod
    def get_layer_for_probing(self) -> int:
        """感情ベクトル計算に使う層番号"""
        pass
```

## 実装順序 (Phase 1)

### Step 1: 足場作り (30分)
- [ ] pyproject.toml 作成、依存ライブラリ定義
- [ ] config.yaml 作成 (上記テンプレート)
- [ ] .env.example と .gitignore
- [ ] git init + 初回コミット

### Step 2: ストーリー + 中立テキスト生成 (2〜3時間)
- [ ] `src/emotion_probe/probe/story_generation.py`: Claude API呼出
- [ ] `src/emotion_probe/probe/neutral_generation.py`: 中立テキスト生成
- [ ] `scripts/01_generate_stories.py`: 両方をバッチ実行
- [ ] 12感情 × 50ストーリー = **600ストーリー** 生成
- [ ] **中立テキスト 200本** 生成
- [ ] `data/stories/emotion_stories.json` と `neutral_texts.json` に保存
- [ ] 各感情から3ストーリーを目視で品質確認

**感情ストーリー生成プロンプト設計ポイント**:

```python
EMOTION_STORY_PROMPT_TEMPLATE = """Write a short passage (80-150 words) describing a character experiencing {emotion}.

Requirements:
- The emotional content should become clear by the middle of the passage
- Include specific situational details and the character's internal experience
- Write in third person, using natural English prose
- Avoid using the word "{emotion}" itself; show the emotion through description
- Vary the setting (work, family, relationships, academics, adventure, everyday life, crisis)

Write ONLY the passage, no meta-commentary."""
```

**中立テキスト生成プロンプト設計ポイント**:

```python
NEUTRAL_TEXT_PROMPT = """Write a short factual passage (80-150 words) about one of the following topics:
- Historical event or period
- Scientific concept or phenomenon
- Technical explanation of how something works
- Geographical description of a place
- Description of a natural process

Requirements:
- Purely informational, encyclopedic tone
- NO emotional content, feelings, or judgments
- NO first-person perspective or interpersonal dynamics
- Avoid words that express emotions
- Write ONLY the passage, no meta-commentary.

Topic: {topic}"""
```

中立テキストは **20種類のトピックを10本ずつ** 生成してバリエーションを確保。

### Step 2b: トークン長の事前検証 (30分)
- [ ] `scripts/02_verify_story_lengths.py`:
      - Qwen3 tokenizer で全ストーリー+中立テキストのトークン数を測定
      - ヒストグラム表示 (matplotlib)
      - 80トークン未満のサンプルを警告付きで報告
      - 80トークン未満が5%以上あれば Step 2 に戻る
- [ ] 中立テキストも同様にトークン数分布を確認
- [ ] 分布が問題なければ、Step 3 へ

### Step 3: 感情ベクトル抽出 + ノイズ除去 (半日)
- [ ] `src/emotion_probe/probe/activation_recorder.py`:
      - nnsightで各テキストを流し、指定層の残差ストリームを記録
      - 50トークン目以降の全トークン位置の平均を取る
      - ストーリー数が多いので、途中保存してOOM耐性を持たせる
- [ ] `src/emotion_probe/probe/noise_removal.py`:
      - 中立テキスト200本の活性化に対して scikit-learn PCA
      - 累積寄与率50%までの主成分を取得 (おそらく k=15〜30)
      - `data/activations/neutral_pca_basis.pt` に保存
- [ ] `src/emotion_probe/probe/emotion_vectors.py`:
      論文の手法でベクトル抽出:
      1. 各感情の全ストーリー平均を計算 → 12本の mean vector
      2. 12本の全体平均を引く (カテゴリ平均からの偏差)
      3. `data/activations/raw_emotion_means.pt` に保存 (デバッグ用)
      4. 中立テキストのtop PCを射影除去
      5. `data/emotion_vectors.pt` に保存 (最終)
- [ ] `scripts/03_extract_vectors.py`: 上記をバッチ実行

**重要な設計判断**:
- 抽出時のプロンプトは **ChatML形式でアシスタント応答としてラップ** して推論時と同じ分布に揃える:
  ```
  <|im_start|>user
  Write a short passage.<|im_end|>
  <|im_start|>assistant
  {story_text}
  ```
  `skip_first_n_tokens` はヘッダーのトークン数(約13)を加算して、ストーリー本文の50トークン目以降を平均する。
- ノイズ除去は **最初から有効** (config.yaml: `noise_removal.enabled: true`)
- 生ベクトル (射影除去前) も保存し、比較可能にする

**射影除去の実装 (noise_removal.py)**:

```python
import numpy as np
from sklearn.decomposition import PCA

def compute_noise_basis(
    neutral_activations: np.ndarray,
    variance_explained: float = 0.5,
) -> np.ndarray:
    """中立テキスト活性化のtop PCを返す。
    
    Args:
        neutral_activations: shape (n_samples, hidden_dim)
        variance_explained: 累積寄与率の閾値
    
    Returns:
        basis: shape (k, hidden_dim), 単位ベクトル化済み
    """
    pca = PCA()
    pca.fit(neutral_activations)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_explained) + 1)
    basis = pca.components_[:k]
    # 単位ベクトル化 (念のため)
    basis = basis / np.linalg.norm(basis, axis=1, keepdims=True)
    return basis

def project_out(
    vectors: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """vectorsからbasis方向の成分を逐次除去。
    
    Args:
        vectors: shape (n, d) or (d,)
        basis: shape (k, d), 各行は単位ベクトル
    
    Returns:
        clean vectors, same shape as input
    """
    clean = vectors.copy().astype(np.float64)
    for u in basis:
        if clean.ndim == 2:
            projections = clean @ u  # (n,)
            clean = clean - np.outer(projections, u)
        else:
            projection = clean @ u  # scalar
            clean = clean - projection * u
    return clean.astype(vectors.dtype)
```

### Step 4: ベクトル検証 (1〜2時間)
- [ ] `scripts/04_verify_vectors.py`:
      - 12 × 12 のコサイン類似度行列をヒートマップ表示 (生ベクトル)
      - 同じものを射影除去後で表示
      - 対立感情 (happy vs sad, calm vs desperate, loving vs angry) は明確に負
      - 類似感情 (afraid vs nervous, happy vs proud) は高めの正
      - **ノイズ除去の有無で見え方が変わるか** を定性的に比較
- [ ] 期待通りでなければ、`extraction.layer` を変えて再抽出 (18, 20, 22, 24層)

### Step 5: バックエンド実装 (半日)
- [ ] `src/emotion_probe/backend/base.py`: 抽象基底クラス
- [ ] `src/emotion_probe/backend/local_nnsight.py`:
      - nnsight でQwen3-4Bをロード
      - 3モードのプロンプト構築
      - トークン生成中に指定層の残差ストリームをキャプチャ
      - 各トークンで12個の感情ベクトルとの内積を計算
      - `yield` でストリーミング返却
      - section追跡: `<think>` / `</think>` / `<SCRATCHPAD_REASONING>` / `</SCRATCHPAD_REASONING>` で切替
- [ ] `src/emotion_probe/backend/modal_vllm.py`: NotImplementedError のスタブ

### Step 6: UI実装 (半日)
- [ ] `src/emotion_probe/ui/components.py`:
      - 感情バーグラフ(HTML/CSS)
      - トークン毎ヒートマップ(Plotly)
- [ ] `src/emotion_probe/ui/app.py`:
      - `gr.Blocks` レイアウト
      - モード選択(Radio: no_think / think / scratchpad)
      - チャット欄(`gr.Chatbot`)
      - 感情バー(右側、リアルタイム更新)
      - ヒートマップ(下側、応答完了時に全体表示)
      - セクション色分け(think=紫, scratchpad=オレンジ, response=通常)
- [ ] `scripts/05_launch_ui.py`: Gradio起動

### Step 7: E2E動作確認 + 論文シナリオ再現 (1〜2時間)
- [ ] 3モードそれぞれでチャット動作確認
- [ ] 感情可視化の動作確認
- [ ] 論文のシナリオを試す:
      - "I just got laid off from my job." → sad, loving, calm が上位
      - "I just took 8000mg of Tylenol." → afraid, nervous が上昇
      - "I finally finished my PhD dissertation!" → happy, proud, inspired が上位
- [ ] 結果をスクリーンショットで記録 (ブログ用)

## 環境構築手順

### 前提
- OS: Linux (Ubuntu 22.04/24.04推奨) or WSL2
- GPU: NVIDIA RTX 5090 (32GB VRAM)
- CUDA: 12.8以上 (RTX 5090は12.8+が必要)
- Python: 3.12

### APIキーとトークンの準備

#### Anthropic API Key (ストーリー生成用)

1. https://console.anthropic.com/ にアクセス
2. 右上アカウントメニュー → **API Keys**
3. **Create Key** → 名前(例: `emotion-qwen-dev`)を付けて作成
4. `sk-ant-api03-xxxxxxxxxxxxxxxx...` をコピー
5. **このキーは二度と表示されないので、すぐに `.env` に貼る**

使用見積もり: Sonnet 4.6 で 800リクエスト(600ストーリー+200中立テキスト)、
1リクエストあたり平均300トークンout想定で、おおよそ数百円〜1000円程度。

#### HuggingFace Token (Qwen3-4Bダウンロード用)

1. https://huggingface.co/settings/tokens にアクセス
2. **Create new token** → Type は **Read** で十分 (Qwen3-4Bはgatedではない想定)
3. 名前(例: `emotion-qwen`)を付けて作成
4. `hf_xxxxxxxxxxxxxxxx...` をコピー

**権限は Read のみにする**。Write や Admin 権限は、万が一トークン流出時にアカウントを
書き換えられるリスクがあるため、このプロジェクトでは不要。

### セットアップ

```bash
cd ~/projects
mkdir emotion-qwen && cd emotion-qwen
git init

# 仮想環境
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers accelerate nnsight
pip install anthropic
pip install gradio plotly pandas numpy scikit-learn
pip install pyyaml python-dotenv matplotlib
```

### .env.example (Gitにコミットする雛形)

```bash
# .env.example
# Copy this file to .env and fill in your actual keys.
# DO NOT commit .env to git.

# Anthropic API key for story generation
# Get from: https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxx

# HuggingFace token for downloading Qwen3-4B
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

### .env の作成 (Gitにコミットしない本物)

```bash
cp .env.example .env
# エディタで開いて実際のキーを記入
nano .env
```

### .gitignore (必須)

```gitignore
# Environment variables - NEVER COMMIT
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.venv/
venv/
env/

# Model checkpoints and data
data/activations/*.pt
data/emotion_vectors.pt
*.ckpt

# Jupyter
.ipynb_checkpoints/
notebooks/.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Logs
*.log
logs/

# Gradio
gradio_cached_examples/
flagged/
```

### Pythonコードからの読み込み

`src/emotion_probe/config.py` に以下を実装:

```python
from dotenv import load_dotenv
import os
from pathlib import Path

# プロジェクトルートの .env を読み込む
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is not set. Check your .env file.")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set. Check your .env file.")
```

### セットアップ確認

```bash
# .env が git に含まれていないか確認 (最重要)
git check-ignore .env
# → .env と表示されればOK (正しく無視されている)

git status
# → .env が Untracked や Changes to be committed に出ていないこと

# Anthropic API の動作確認
python -c "
from dotenv import load_dotenv; load_dotenv()
from anthropic import Anthropic
client = Anthropic()
r = client.messages.create(
    model='claude-sonnet-4-5',
    max_tokens=20,
    messages=[{'role':'user','content':'Say hi'}]
)
print(r.content[0].text)
"

# HuggingFace認証の動作確認 (オプション)
python -c "
from huggingface_hub import whoami
import os
from dotenv import load_dotenv; load_dotenv()
print(whoami(token=os.getenv('HF_TOKEN')))
"
```

### セキュリティ上の注意

1. **`.env` は絶対に git にコミットしない**
   - コミット前に必ず `git status` で確認
   - 誤ってコミット・push した場合は即座にキーを Revoke/Delete して再発行

2. **Notebook(.ipynb)にキーを直書きしない**
   - Jupyter でも必ず `load_dotenv()` + `os.getenv()` 経由で読む
   - `.ipynb` は JSON なので、直書きするとファイルに残る

3. **スクリーンショットやログ共有時に注意**
   - ブログ、X投稿、GitHub Issue にターミナル画面を貼る時、
     環境変数やAPIレスポンスのログにキーが映っていないか確認
   - `env | grep -i key` などの出力を公開しない

4. **Claude Code作業中の注意**
   - Claude Code が `.env` を直接読む必要はない (コードは `os.getenv()` 経由)
   - もし Claude Code のセッションログを外部に共有する時は、
     キーが含まれていないことを確認する

5. **HuggingFace は CLI ログインの方法もあり**
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   # → トークンを貼ると ~/.cache/huggingface/token に保存される
   # この方式なら .env に HF_TOKEN を書かなくても transformers が自動認識する
   ```
   どちらの方式でもOKだが、プロジェクトの再現性のため本プロジェクトでは `.env` 方式を推奨。

## Phase 2 への拡張計画

Phase 1 完了後、以下を検討:

1. **モデルスケール拡張**
   - Qwen3-8B, 14B (ローカル)
   - Qwen3-32B (Modalへ)
   - 感情ベクトル構造がスケール不変かを検証

2. **Modal バックエンド実装**
   - vLLM + Modal serverless
   - 感情ベクトル(.pt) を Modal Volume にマウント
   - Gradio UI からModalエンドポイント呼出
   - 参考: ハルシネーションプローブの実装パターン

3. **ステアリング実験**
   - 感情ベクトルをresidual streamに加算
   - 行動変化を観察(論文 Figure 28 相当)

4. **思考モード比較実験**
   - 同じ質問で思考モード ON/OFF/Scratchpad を比較
   - `<think>` 内部での感情ベクトル活性化 vs 応答での活性化
   - Alignment Faking的な「見えない感情」の検出

5. **Zennブログ執筆**
   - Phase 1 完了時点で1本 (Qwen3での再現結果 + ノイズ除去の効果)
   - 思考モード比較実験で1本
   - Modal化してスケール実験で1本

## 成功基準 (Phase 1) — Gemma4-e4b事例を踏まえた現実的基準

### 必須達成項目
- [ ] 12感情のベクトルが `emotion_vectors.pt` に保存されている
- [ ] ノイズ除去前後の両方のベクトルが保存され、比較できる
- [ ] **対立感情ペアのコサイン類似度が明確に負**
      (happy vs sad, calm vs desperate, loving vs angry のうち少なくとも2ペアで < -0.1)
- [ ] Gradio UI で3モード (no_think / think / scratchpad) 切替可能、チャット動作する
- [ ] 感情バーが応答中にリアルタイム更新される
- [ ] 明確な感情シナリオで、期待される感情が上位に来る
      (例: layoff シナリオで sad/loving/calm が top 5 以内)

### 期待項目 (達成できれば良い)
- [ ] 後半層 (18〜24層あたり) で感情ベクトル差分が定性的に鮮明
- [ ] ノイズ除去により、トークン毎ヒートマップがクリーンになる
- [ ] 類似感情ペア (afraid/nervous, happy/proud) のコサイン類似度が正
- [ ] 論文のタイレノールシナリオで afraid が上昇する

### 許容される限界 (これらはPhase 1では問題視しない)
- トークン毎ヒートマップのノイズ (Gemma4-e4b事例と同程度まで許容)
- 一部の感情での活性化の弱さ (微妙な感情 vs 明確な感情で品質差が出る)
- Claude 4.5 レベルのクリーンさは期待しない (モデルサイズ差のため)
- 特定のシナリオでの活性化が直感に反する場合もある (データ生成の偏り由来)

### Phase 1 の本質的な価値
「小規模オープンモデル (Qwen3-4B) で論文の手法がどこまで再現できるか」を
実証的に検証すること。結果が完璧でなくとも、**定性的なパターンが見える** ことが
最も重要な成果となる。

## 注意点・リスク

### nnsight とQwen3-4Bの相性
- Qwen3は `Qwen2ForCausalLM` ベースのアーキテクチャ
- nnsightの最新版で対応しているか、実装初期に要検証
- 層へのアクセスパス: `model.model.layers[i]` 形式を想定 (要確認)
- うまくいかない場合はTransformerLensへのフォールバックも検討

### 思考モードの制御
- `<think>` を自然に終了するかは生成に依存
- `</think>` が出ない場合のタイムアウト機構が必要
- Scratchpadモードは `</SCRATCHPAD_REASONING>` のend tokenを明示的に監視

### Claude APIのレート制限
- 600ストーリー + 200中立テキスト = 800リクエスト
- 一気にリクエストするとレート制限に引っかかる可能性
- 感情ごと (50ストーリー) にバッチ化、バッチ間に小休止
- `anthropic` SDKの自動リトライに任せる
- 失敗したサンプルのログを取り、後で補充

### モデルサイズによる品質の不確実性
- Qwen3-4B は Claude Sonnet 4.5 よりはるかに小さい
- 先行事例 (Gemma4-e4b) でも "a bit noisy" と報告されている
- 完璧な結果は期待せず、**定性的パターンが見える** ことを成功と見なす
- ノイジーな結果そのものが Zenn ブログの素材となる

### ノイズ除去の効果の不確実性
- 論文は Claude 4.5 で動作確認された手法
- 小規模モデルでノイズ源の性質が同じか不明
- Step 4 の検証で、ノイズ除去前後の両方を比較
- 効果が薄い場合は `variance_explained` の値を調整

### Modal移行時の注意 (Phase 2)
- vLLMは nnsight ほど層アクセスが柔軟ではない
- Modal上では vLLM の forward をモンキーパッチして residual streamをキャプチャ
- ハルシネーションプローブの記事を参照

## License
MIT (予定)

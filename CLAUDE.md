# Claude Code 起動時プロンプト

以下を `CLAUDE.md` としてプロジェクトルートに配置しておくと、
Claude Code が起動時にコンテキストとして自動読込します。

---

# CLAUDE.md

You are working on **Emotion Probe for Qwen3**, a research reproduction project
for Anthropic's 2026 "Emotion Concepts" paper.

## プロジェクトの読み方

まず `SPEC.md` を読んで、全体の設計を把握してください。
これがこのプロジェクトの正典です。

## 作業スタイル

- **実験コードであり、商用コードではない** — 可読性と再現性を優先
- **日本語でコミュニケーション** (ユーザーは日本人研究者)
- コメントは英語でもOK
- 変数名・関数名は英語

## 重要な制約

1. **GPU 実行は慎重に**
   - RTX 5090 (32GB VRAM) を使用
   - Qwen3-4B Dense を bfloat16 でロード (約10GB VRAM)
   - OOMを避けるため、長時間実行は感情単位で区切って保存

2. **API呼出はバッチ化**
   - Claude API (ストーリー生成) はレート制限あり
   - 感情ごと50ストーリーをまとめて1リクエストせず、適度に分割

3. **層アクセスパスの検証が必要**
   - nnsightでQwen3-4Bの `model.model.layers[i]` アクセスを最初に確認
   - うまくいかなければ `print(lm.model)` で構造を確認してから進める

4. **APIキー・トークンの取扱い (最重要)**
   - `.env` ファイルは**絶対に git にコミットしない**
   - `.env.example` は雛形としてコミットしてOK
   - コード内では `os.getenv()` 経由でのみキーを参照
   - 実装中に `.env` の中身を直接ターミナルに出力したり、チャットに貼らない
   - `.gitignore` に `.env` が含まれていることを各コミット前に確認

## Git フロー

- Step ごとにコミット (Step 1 完了 → commit → Step 2 開始)
- コミットメッセージ: `[Step N] 内容` 形式 (日本語可)
- Phase 1 完了時点で v0.1.0 タグを打つ

## 実装開始時の手順

1. `SPEC.md` を熟読
2. "Step 1: 足場作り" から順番に進める
3. 各Stepの完了確認 (check-list) をユーザーに報告
4. 次のStepに進む前にコミット

## 特に注意するポイント

- **感情ベクトル抽出時はモード非依存** (平文ストーリー)
- **推論時のみ3モード切替** (no_think / think / scratchpad)
- セクション判定 (think / scratchpad / response) は文字列マッチで十分
- `50トークン目以降` は論文準拠 (それより短いテキストはスキップかwarn)

## ユーザーからのフィードバック受入

実装中にユーザーから「ここは違う」「この方が良い」と来た場合、
SPEC.md を更新してから実装を変更すること。SPECが乖離するとカオスになる。

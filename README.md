# PageSnap

教科書・問題集のページをめくりながら撮った動画から、ページが止まっている瞬間だけを画像として切り出し、
Gemini API に「なぜこうなるか」を解説させて `summary.md` にまとめるコマンドラインツールです。

## セットアップ

```bash
pip install -r requirements.txt
```

Gemini API キーを [Google AI Studio](https://aistudio.google.com/apikey) で発行し、環境変数に設定します。

```bash
# macOS / Linux
export GEMINI_API_KEY="取得したキー"

# Windows (PowerShell)
$env:GEMINI_API_KEY="取得したキー"
```

## 使い方

```bash
python pagesnap.py --input 数学B_10月8日.mp4 --subject 数学B
```

動画と同じフォルダに `数学B_10月8日_20261008_output/` ができ、中に次のファイルが入ります。

- `page_01.jpg`, `page_02.jpg`, … 抽出したページ画像
- `summary.md` 目次と、各ページの画像と解説

同じ日に2回実行すると、前回の結果を上書きせず `..._output_2/` が作られます。

### 主なオプション

| オプション | 説明 |
|---|---|
| `--subject` | 科目名。フォルダ名とプロンプトに使います |
| `--no-gemini` | 画像の抽出だけ行います（API キー不要） |
| `--model` | Gemini のモデル名（既定: `gemini-flash-latest`。環境変数 `PAGESNAP_GEMINI_MODEL` でも指定可） |
| `--output-dir` | 出力フォルダを作る場所 |
| `--still-threshold` | 静止とみなす差分の上限（既定: 12） |
| `--min-still-sec` | この秒数以上止まっていればページとみなす（既定: 0.5） |
| `--dedupe-distance` | 重複とみなす画像ハッシュ差（既定: 36、0 で無効） |

### ページ数が合わないとき

- 同じページが何枚も出る → `--still-threshold 18` や `--dedupe-distance 50`
- ページが抜ける → `--min-still-sec 0.3` や `--still-threshold 8`

### 撮影のコツ

- 1ページにつき 1 秒ほど手を止める
- めくり終わったら指をページから離す（指が映ったフレームは選ばれにくくしていますが、ゼロにはできません）

## 仕組み

1. 動画を 1 秒 4 フレームの間隔で読み、縮小・ぼかしした画像同士の差分を計算
2. 差分が小さい状態が一定時間続いた区間を「静止区間」とする
3. 各区間の中央付近で最もピントが合ったフレームを代表として選ぶ
4. dHash（16×16）が近い画像を重複として除く
5. 各画像を Gemini に送り、解説を `summary.md` に1ページずつ追記（途中で止まってもそこまでは残る）

一時的な API エラー（429 / 5xx / 通信断）は自動で再試行し、1ページ失敗しても残りのページは続行します。

## Claude Code のスキルとして使う

このフォルダごと `~/.claude/skills/pagesnap/` に置くと、`SKILL.md` によって
「この動画を pagesnap で処理して」と頼むだけで実行されるようになります。

## テスト

```bash
pip install -r requirements-dev.txt
python -m pytest
```

テストは合成した「ページをめくる動画」を使うので、API キーや実際の動画は不要です。

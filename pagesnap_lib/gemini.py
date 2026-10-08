"""Gemini API への画像送信と解説生成。"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

DEFAULT_MODEL = "gemini-flash-latest"
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

BASE_PROMPT = (
    "この画像は教科書または問題集のページです。"
    "書かれている問題や内容を読み取り、重要なポイントと、"
    "問題があれば解答の根拠・なぜそうなるかを説明してください。"
)

FORMAT_RULES = (
    "\n\n# 出力ルール\n"
    "- 日本語の Markdown で書く\n"
    "- 見出しは「###」以下のレベルだけを使う（まとめファイルの目次を崩さないため）\n"
    "- 読み取れない・判別できない箇所は推測で埋めず「読み取れません」と書く\n"
    "- 問題が複数あるときは問題番号ごとに分けて説明する"
)

# 再試行する HTTP ステータス（レート制限・一時的なサーバーエラー）
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
# 続けても全ページ失敗するだけのエラー（キー不正・権限なし・モデル名違い）
FATAL_CODES = {401, 403, 404}


class GeminiFatalError(RuntimeError):
    """これ以上ページを送っても無駄なエラー。"""


class GeminiPageError(RuntimeError):
    """そのページだけの失敗。次のページは続行してよい。"""


def get_api_key() -> str | None:
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def build_prompt(subject: str | None) -> str:
    prompt = BASE_PROMPT
    if subject:
        prompt = f"科目は「{subject}」です。" + prompt
    return prompt + FORMAT_RULES


def make_client(api_key: str) -> Any:
    try:
        from google import genai
    except ImportError as e:
        raise GeminiFatalError(
            "google-genai がインストールされていません。`pip install -r requirements.txt` を実行してください。"
        ) from e
    return genai.Client(api_key=api_key)


def explain_page(
    client: Any,
    model: str,
    image_bytes: bytes,
    subject: str | None,
    max_attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """1ページ分の画像を送り、解説テキストを返す。

    一時的なエラーは指数バックオフで再試行する。
    失敗時は GeminiPageError（このページだけ）か GeminiFatalError（全体を止める）を投げる。
    """
    from google.genai import errors, types

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        build_prompt(subject),
    ]
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(model=model, contents=contents)
            return _extract_text(response)
        except errors.APIError as e:
            last_error = e
            code = getattr(e, "code", None)
            message = str(e)
            if code in FATAL_CODES or (code == 400 and "API key" in message):
                raise GeminiFatalError(_describe_fatal(code, model, message)) from e
            if code not in RETRYABLE_CODES:
                raise GeminiPageError(f"Gemini API エラー ({code}): {message}") from e
        except (OSError, TimeoutError) as e:
            # ネットワーク切断など
            last_error = e
        except Exception as e:  # httpx の接続エラー等。SDK 内部の例外型に依存しないため広めに取る
            if type(e).__module__.startswith(("httpx", "httpcore")):
                last_error = e
            else:
                raise
        if attempt < max_attempts:
            sleep(delay)
            delay = min(delay * 2, 60.0)
    raise GeminiPageError(f"{max_attempts} 回試しても失敗しました: {last_error}")


def _extract_text(response: Any) -> str:
    # 安全フィルタでブロックされた場合などは response.text が None / 例外になる
    try:
        text = response.text
    except Exception:
        text = None
    if text and text.strip():
        return text.strip()

    reason = None
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        reason = f"ブロックされました ({feedback.block_reason})"
    else:
        candidates = getattr(response, "candidates", None) or []
        if candidates and getattr(candidates[0], "finish_reason", None):
            reason = f"終了理由: {candidates[0].finish_reason}"
    raise GeminiPageError("Gemini から解説が返ってきませんでした" + (f"（{reason}）" if reason else ""))


def _describe_fatal(code: int | None, model: str, message: str) -> str:
    if code == 404:
        return (
            f"モデル「{model}」が見つかりません。--model か環境変数 PAGESNAP_GEMINI_MODEL で"
            f"利用可能なモデル名（例: gemini-2.5-flash）を指定してください。\n詳細: {message}"
        )
    return f"Gemini API キーが無効か、権限がありません（{code}）。GEMINI_API_KEY を確認してください。\n詳細: {message}"

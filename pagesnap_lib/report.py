"""summary.md の生成。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PageResult:
    number: int
    image_name: str
    timestamp: float
    explanation: str | None = None  # None = まだ / 送っていない
    error: str | None = None


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def anchor(page: PageResult) -> str:
    return f"page-{page.number:02d}"


def render_summary(
    title: str,
    source_name: str,
    subject: str | None,
    created: str,
    pages: list[PageResult],
    note: str | None = None,
) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- 元動画: `{source_name}`")
    if subject:
        lines.append(f"- 科目: {subject}")
    lines.append(f"- 作成日時: {created}")
    lines.append(f"- ページ数: {len(pages)}")
    if note:
        lines += ["", f"> {note}"]
    lines += ["", "## 目次", ""]
    for p in pages:
        mark = " ⚠️" if p.error else ""
        lines.append(f"- [ページ {p.number}（{format_timestamp(p.timestamp)}）](#{anchor(p)}){mark}")
    for p in pages:
        lines += [
            "",
            "---",
            "",
            f'<a id="{anchor(p)}"></a>',
            "",
            f"## ページ {p.number}（{format_timestamp(p.timestamp)}）",
            "",
            f"![ページ {p.number}]({p.image_name})",
            "",
        ]
        if p.explanation:
            lines.append(p.explanation)
        elif p.error:
            lines.append(f"> ⚠️ 解説を生成できませんでした: {p.error}")
        else:
            lines.append("> （解説なし）")
    return "\n".join(lines) + "\n"


def write_summary(path: Path, content: str) -> None:
    # 途中で落ちても壊れたファイルが残らないよう、一時ファイルに書いてから置き換える
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

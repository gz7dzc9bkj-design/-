#!/usr/bin/env python3
"""PageSnap: ページをめくる動画から静止ページを切り出し、Gemini に解説させる。

使い方:
    python pagesnap.py --input 数学B_10月8日.mp4 --subject 数学B
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from pagesnap_lib import detection, gemini, report, video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ページをめくる動画から静止ページを切り出し、Gemini で解説を生成します。"
    )
    p.add_argument("--input", "-i", required=True, help="動画ファイルのパス")
    p.add_argument("--subject", "-s", help="科目名（出力フォルダ名とプロンプトに使用）")
    p.add_argument("--output-dir", help="出力フォルダを作る場所（既定: 動画と同じフォルダ）")
    p.add_argument("--no-gemini", action="store_true", help="ページ画像の抽出だけ行い、Gemini には送らない")
    p.add_argument(
        "--model",
        default=os.environ.get("PAGESNAP_GEMINI_MODEL", gemini.DEFAULT_MODEL),
        help=f"Gemini のモデル名（既定: {gemini.DEFAULT_MODEL}、環境変数 PAGESNAP_GEMINI_MODEL でも指定可）",
    )
    tuning = p.add_argument_group("検出の調整（ページ数が合わないとき）")
    tuning.add_argument("--sample-fps", type=float, default=4.0, help="1秒あたりに調べるフレーム数（既定: 4）")
    tuning.add_argument(
        "--still-threshold",
        type=float,
        default=12.0,
        help="静止とみなす差分の上限（既定: 12）。ページが取れすぎる→下げる、取れなさすぎる→上げる",
    )
    tuning.add_argument(
        "--min-still-sec", type=float, default=0.5, help="この秒数以上止まっていればページとみなす（既定: 0.5）"
    )
    tuning.add_argument(
        "--dedupe-distance",
        type=int,
        default=36,
        help="重複とみなすハッシュ差の上限（256ビット中、既定: 36）。0 で重複除去を無効化",
    )
    args = p.parse_args(argv)
    if args.sample_fps <= 0:
        p.error("--sample-fps は正の数にしてください")
    if args.min_still_sec < 0:
        p.error("--min-still-sec は 0 以上にしてください")
    return args


def sanitize_filename(name: str) -> str:
    """Windows/macOS でフォルダ名に使えない文字を置き換える。"""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or "pagesnap"


def make_output_dir(input_path: Path, subject: str | None, base: Path | None, now: datetime) -> Path:
    stem = input_path.stem
    if subject and subject not in stem:
        stem = f"{subject}_{stem}"
    name = sanitize_filename(f"{stem}_{now:%Y%m%d}_output")
    parent = base or input_path.parent
    out = parent / name
    # 前回の結果に上書きすると、古い page_XX.jpg が混ざるので連番で別フォルダにする
    n = 2
    while out.exists():
        out = parent / f"{name}_{n}"
        n += 1
    out.mkdir(parents=True)
    return out


def find_pages(args: argparse.Namespace, input_path: Path) -> list[video.SampledFrame]:
    """動画を解析し、ページとして保存すべきフレームを返す。"""
    print(f"動画を解析しています: {input_path.name}")
    frames, _ = video.sample_frames(input_path, args.sample_fps)
    print(f"  長さ 約{report.format_timestamp(frames[-1].timestamp)}、{len(frames)} フレームを調べました")

    min_length = max(2, math.ceil(args.min_still_sec * args.sample_fps))
    periods = detection.segment_still_periods(
        [f.small for f in frames], threshold=args.still_threshold, min_length=min_length
    )
    if not periods:
        return []
    reps = [frames[i] for i in detection.pick_representatives(periods, [f.sharpness for f in frames])]
    print(f"  静止区間: {len(periods)} 個")

    if args.dedupe_distance > 0:
        # dHash は 17x16 に縮小してから計算するので、縮小済み画像で判定しても結果はほぼ同じ
        kept = detection.dedupe_by_hash([f.small for f in reps], max_distance=args.dedupe_distance)
        if len(kept) < len(reps):
            print(f"  重複ページを {len(reps) - len(kept)} 枚除きました")
        reps = [reps[i] for i in kept]
    return reps


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        print(f"エラー: 動画ファイルが見つかりません: {input_path}", file=sys.stderr)
        return 1

    # 重い動画処理の前に、APIキーの有無を確認しておく
    client = None
    if not args.no_gemini:
        api_key = gemini.get_api_key()
        if not api_key:
            print(
                "エラー: Gemini API キーが設定されていません。\n"
                "  環境変数 GEMINI_API_KEY に設定するか、抽出だけなら --no-gemini を付けてください。",
                file=sys.stderr,
            )
            return 1
        try:
            client = gemini.make_client(api_key)
        except gemini.GeminiFatalError as e:
            print(f"エラー: {e}", file=sys.stderr)
            return 1

    try:
        pages = find_pages(args, input_path)
    except video.VideoError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    if not pages:
        print(
            "ページを検出できませんでした。カメラが動き続けていた可能性があります。\n"
            "  --still-threshold を上げる（例: 18）か、--min-still-sec を下げて再実行してください。",
            file=sys.stderr,
        )
        return 2

    now = datetime.now()
    out_dir = make_output_dir(
        input_path, args.subject, Path(args.output_dir).expanduser() if args.output_dir else None, now
    )
    width = max(2, len(str(len(pages))))
    results: list[report.PageResult] = []
    image_paths: list[Path] = []
    by_index = {f.frame_index: f for f in pages}
    try:
        for frame_index, img in video.iter_frames_at(input_path, list(by_index)):
            n = len(results) + 1
            name = f"page_{n:0{width}d}.jpg"
            video.save_jpeg(out_dir / name, img)
            image_paths.append(out_dir / name)
            results.append(report.PageResult(number=n, image_name=name, timestamp=by_index[frame_index].timestamp))
    except video.VideoError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    if not results:
        print("エラー: ページ画像を動画から取り出せませんでした。", file=sys.stderr)
        return 1
    print(f"{len(results)} ページを保存しました: {out_dir}")

    summary_path = out_dir / "summary.md"
    title = input_path.stem
    if args.subject and args.subject not in title:
        title = f"{args.subject} {title}"
    title += " 解説まとめ"
    created = now.strftime("%Y-%m-%d %H:%M")
    note = "Gemini には送信していません（--no-gemini）。" if args.no_gemini else None

    def save(note: str | None = note) -> None:
        report.write_summary(
            summary_path, report.render_summary(title, input_path.name, args.subject, created, results, note)
        )

    save()
    exit_code = 0
    if client is not None:
        for page, image_path in zip(results, image_paths):
            data = image_path.read_bytes()
            print(f"  ページ {page.number}/{len(results)} を Gemini に送信中...", flush=True)
            try:
                page.explanation = gemini.explain_page(client, args.model, data, args.subject)
            except gemini.GeminiPageError as e:
                page.error = str(e)
                print(f"    失敗: {e}", file=sys.stderr)
                exit_code = 3
            except gemini.GeminiFatalError as e:
                print(f"エラー: {e}", file=sys.stderr)
                save("Gemini への送信が途中で止まりました。画像は保存済みです。")
                return 1
            except KeyboardInterrupt:
                save("中断されました。ここまでの解説は保存済みです。")
                raise
            # 1ページごとに保存して、途中で落ちてもそれまでの解説を残す
            save()

    print(f"完了: {summary_path}")
    if exit_code:
        print("一部のページで解説を取得できませんでした（summary.md に ⚠️ で表示）。", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

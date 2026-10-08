"""静止区間検出・代表フレーム選定・画像ハッシュ・重複除去の純粋関数群。

どれも動画やAPIに依存しないので、numpy 配列だけでテストできる。
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

# 差分計算用の縮小幅とぼかし。手持ち撮影の数pxの揺れで文字の輪郭が差分に出ないよう、
# かなり小さく・強めにぼかす（ページめくりの差分はこれでも十分大きく残る）。
DIFF_WIDTH = 160
DIFF_BLUR = 9


def preprocess_for_diff(image: np.ndarray, width: int = DIFF_WIDTH) -> np.ndarray:
    """差分計算用に、縮小・グレースケール化・ぼかしをかける。

    フル解像度のまま差分を取ると、手ブレやセンサーノイズで静止判定が不安定になるため。
    """
    gray = _to_gray(image)
    h, w = gray.shape[:2]
    if w > width:
        new_h = max(1, round(h * width / w))
        gray = cv2.resize(gray, (width, new_h), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (DIFF_BLUR, DIFF_BLUR), 0)


def frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    """2枚の画像の平均絶対差（0〜255）。"""
    if a.shape != b.shape:
        raise ValueError(f"画像サイズが一致しません: {a.shape} vs {b.shape}")
    return float(np.mean(cv2.absdiff(a, b)))


def sharpness(image: np.ndarray) -> float:
    """ラプラシアン分散によるピントの鋭さ。大きいほどくっきりしている。"""
    return float(cv2.Laplacian(_to_gray(image), cv2.CV_64F).var())


def segment_still_periods(
    frames: Sequence[np.ndarray],
    threshold: float = 12.0,
    min_length: int = 2,
    drift_factor: float = 2.5,
) -> list[tuple[int, int]]:
    """ピクセル差分が閾値以下で続く区間を「静止区間」としてまとめる。

    Args:
        frames: preprocess_for_diff 済みの画像列（全て同じサイズ）。
        threshold: 隣接フレーム間の平均絶対差がこれ以下なら「静止」とみなす。
        min_length: これ未満のフレーム数しか続かない区間は捨てる（めくり途中の一瞬の停止対策）。
        drift_factor: 区間の先頭フレームとの差が threshold * drift_factor を超えたら区間を切る。
            ゆっくりページをめくった場合など、隣接差分は小さいのに絵が変わっていく状況への対策。

    Returns:
        (開始インデックス, 終了インデックス) のリスト。両端を含む。
    """
    if min_length < 1:
        raise ValueError("min_length は 1 以上にしてください")
    n = len(frames)
    if n == 0:
        return []

    periods: list[tuple[int, int]] = []
    start = 0
    for i in range(1, n):
        still = frame_diff(frames[i - 1], frames[i]) <= threshold
        if still and drift_factor > 0:
            still = frame_diff(frames[start], frames[i]) <= threshold * drift_factor
        if not still:
            if i - start >= min_length:
                periods.append((start, i - 1))
            start = i
    if n - start >= min_length:
        periods.append((start, n - 1))
    return periods


def pick_representatives(
    still_periods: Sequence[tuple[int, int]],
    sharpness_scores: Sequence[float] | None = None,
) -> list[int]:
    """各静止区間から代表フレームのインデックスを1つずつ選ぶ。

    基本は区間の中間。sharpness_scores を渡すと、区間の中央 50% の中で最もくっきりした
    フレームを選ぶ（端は指の出入りやブレが残りやすいので避ける）。
    """
    reps: list[int] = []
    for start, end in still_periods:
        if end < start:
            raise ValueError(f"不正な区間です: {(start, end)}")
        mid = (start + end) // 2
        if sharpness_scores is None:
            reps.append(mid)
            continue
        length = end - start + 1
        margin = length // 4
        lo, hi = start + margin, end - margin
        best = max(
            range(lo, hi + 1),
            key=lambda i: (sharpness_scores[i], -abs(i - mid)),
        )
        reps.append(best)
    return reps


def compute_dhash(image: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """差分ハッシュ（dHash）を計算する。

    imagehash.dhash と同じ考え方：(hash_size+1) x hash_size に縮小し、
    横に隣り合う画素の大小をビットにする。戻り値は長さ hash_size**2 の bool 配列。
    8x8 だと文字ばかりのページ同士が似たハッシュになりやすいので、既定は 16x16。
    """
    gray = _to_gray(image)
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        raise ValueError("ハッシュの長さが一致しません")
    return int(np.count_nonzero(a != b))


def dedupe_by_hash(
    images: Sequence[np.ndarray],
    max_distance: int = 36,
    hash_size: int = 16,
) -> list[int]:
    """dHash が近い画像同士を重複とみなして間引き、残す画像のインデックスを返す。

    先に出てきた画像を残す。max_distance は hash_size**2 ビット中の許容差。
    別ページを誤って消すとその内容が丸ごと失われるので、既定値は控えめにしている。
    """
    kept: list[int] = []
    kept_hashes: list[np.ndarray] = []
    for i, img in enumerate(images):
        h = compute_dhash(img, hash_size)
        if any(hamming_distance(h, k) <= max_distance for k in kept_hashes):
            continue
        kept.append(i)
        kept_hashes.append(h)
    return kept


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"対応していない画像形式です: shape={image.shape}")

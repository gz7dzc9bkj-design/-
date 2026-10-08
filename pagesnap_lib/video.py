"""動画からのフレーム取得と画像保存。"""

from __future__ import annotations

import math
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from .detection import preprocess_for_diff, sharpness

# FPS が取れない動画（一部のスマホ動画や壊れたメタデータ）のときの仮定値
FALLBACK_FPS = 30.0


class VideoError(RuntimeError):
    pass


@dataclass
class SampledFrame:
    frame_index: int  # 元動画でのフレーム番号
    timestamp: float  # 秒
    small: np.ndarray  # 差分計算用の縮小グレー画像
    sharpness: float


@dataclass
class VideoInfo:
    fps: float
    frame_count: int  # 不明なら 0


@contextmanager
def open_video(path: Path) -> Iterator[cv2.VideoCapture]:
    """動画を開く。Windows で日本語パスが開けない場合は一時的に ASCII 名へコピーして開く。"""
    cap = cv2.VideoCapture(str(path))
    tmp_dir = None
    if not cap.isOpened():
        cap.release()
        tmp_dir = tempfile.mkdtemp(prefix="pagesnap_")
        tmp_path = Path(tmp_dir) / ("input" + path.suffix.lower())
        shutil.copyfile(path, tmp_path)
        cap = cv2.VideoCapture(str(tmp_path))
    try:
        if not cap.isOpened():
            raise VideoError(
                f"動画を開けませんでした: {path}\n"
                "対応していない形式の可能性があります（HEVC の .mov なら mp4/H.264 に書き出して再試行してください）。"
            )
        yield cap
    finally:
        cap.release()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def get_info(cap: cv2.VideoCapture) -> VideoInfo:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0 or fps > 1000:
        fps = FALLBACK_FPS
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return VideoInfo(fps=fps, frame_count=max(count, 0))


def sample_frames(path: Path, sample_fps: float) -> tuple[list[SampledFrame], VideoInfo]:
    """動画を先頭から順に読み、sample_fps 間隔でフレームを取得する。

    CAP_PROP_POS_FRAMES でのシークはコーデックによって位置がずれるので使わず、
    grab() で読み飛ばす。メモリ節約のため、フル解像度の画像は保持しない。
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps は正の数にしてください")
    frames: list[SampledFrame] = []
    with open_video(path) as cap:
        info = get_info(cap)
        step = info.fps / sample_fps
        next_pick = 0.0
        idx = 0
        while cap.grab():
            if idx + 1e-6 >= next_pick:
                ok, img = cap.retrieve()
                if ok and img is not None:
                    frames.append(
                        SampledFrame(
                            frame_index=idx,
                            timestamp=idx / info.fps,
                            small=preprocess_for_diff(img),
                            sharpness=sharpness(img),
                        )
                    )
                next_pick += step
            idx += 1
    if not frames:
        raise VideoError(f"動画からフレームを1枚も読み取れませんでした: {path}")
    return frames, info


def iter_frames_at(path: Path, frame_indices: Sequence[int]) -> Iterator[tuple[int, np.ndarray]]:
    """指定したフレーム番号のフル解像度画像を、先頭から順に読みながら1枚ずつ返す。

    4K 動画だと1枚 25MB 近くあるので、まとめて保持せず呼び出し側ですぐ保存させる。
    """
    wanted = set(frame_indices)
    if not wanted:
        return
    last = max(wanted)
    with open_video(path) as cap:
        idx = 0
        while idx <= last and cap.grab():
            if idx in wanted:
                ok, img = cap.retrieve()
                if ok and img is not None:
                    yield idx, img
            idx += 1


def save_jpeg(path: Path, image: np.ndarray, quality: int = 92) -> bytes:
    """JPEG で保存し、そのバイト列を返す。

    cv2.imwrite は Windows で日本語パスだと何も言わずに失敗するので、
    imencode してから Python 側で書き込む。
    """
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VideoError(f"画像のエンコードに失敗しました: {path}")
    data = buf.tobytes()
    path.write_bytes(data)
    return data

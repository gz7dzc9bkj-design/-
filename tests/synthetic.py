"""テスト用の「ページをめくる動画」を合成する。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

W, H = 480, 640


def make_page(seed: int) -> np.ndarray:
    """文字が並んだ、ページごとに中身が違う画像。"""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 245, np.uint8)
    cv2.putText(img, f"PAGE {seed}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    y = 110
    while y < H - 20:
        x = 30
        while x < W - 60:
            word = int(rng.integers(20, 90))
            if rng.random() > 0.15:
                cv2.rectangle(img, (x, y), (x + word, y + 10), (40, 40, 40), -1)
            x += word + 12
        y += int(rng.integers(22, 40))
    # 図形も入れて、ページごとの違いを大きくする
    cx, cy = int(rng.integers(100, W - 100)), int(rng.integers(200, H - 100))
    cv2.circle(img, (cx, cy), int(rng.integers(30, 80)), (0, 0, 200), 4)
    return img


def write_flip_video(
    path: Path,
    pages: list[np.ndarray],
    fps: int = 30,
    still_sec: float = 1.5,
    flip_sec: float = 0.6,
    noise: float = 2.0,
    jitter: int = 1,
    seed: int = 0,
) -> None:
    """各ページを still_sec 秒静止→ flip_sec 秒かけて横スライドで次のページへ、を繰り返す。"""
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    assert writer.isOpened()

    def emit(img: np.ndarray) -> None:
        # 手持ち撮影っぽいノイズと揺れ
        shift = rng.integers(-jitter, jitter + 1, size=2)
        m = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        frame = cv2.warpAffine(img, m, (W, H), borderMode=cv2.BORDER_REPLICATE)
        frame = np.clip(frame + rng.normal(0, noise, frame.shape), 0, 255).astype(np.uint8)
        writer.write(frame)

    for i, page in enumerate(pages):
        for _ in range(int(still_sec * fps)):
            emit(page)
        if i + 1 < len(pages):
            nxt = pages[i + 1]
            steps = int(flip_sec * fps)
            for s in range(1, steps + 1):
                off = int(W * s / steps)
                frame = np.empty_like(page)
                frame[:, : W - off] = page[:, off:]
                frame[:, W - off :] = nxt[:, :off]
                # 指（肌色の楕円）が横切る
                cv2.ellipse(frame, (W - off, H // 2), (40, 90), 0, 0, 360, (120, 160, 220), -1)
                emit(frame)
    writer.release()

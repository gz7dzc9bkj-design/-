"""合成動画を使った通し動作テスト（Gemini は偽物に差し替える）。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import pagesnap
from pagesnap_lib import gemini
from tests.synthetic import make_page, write_flip_video


@pytest.fixture
def flip_video(tmp_path: Path) -> Path:
    # 日本語ファイル名でも動くことを確認する
    path = tmp_path / "数学B_10月8日.avi"
    pages = [make_page(i) for i in range(6)]
    write_flip_video(path, pages)
    return path


def test_extract_only(flip_video: Path, tmp_path: Path):
    out_base = tmp_path / "out"
    code = pagesnap.main(["--input", str(flip_video), "--subject", "数学B", "--no-gemini", "--output-dir", str(out_base)])
    assert code == 0
    (out_dir,) = out_base.iterdir()
    assert out_dir.name.startswith("数学B_10月8日_") and out_dir.name.endswith("_output")
    images = sorted(out_dir.glob("page_*.jpg"))
    assert len(images) == 6
    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "## 目次" in summary and "page_06.jpg" in summary

    # 2回目は別フォルダになり、前回の結果を上書きしない
    assert pagesnap.main(["--input", str(flip_video), "--no-gemini", "--output-dir", str(out_base)]) == 0
    assert len(list(out_base.iterdir())) == 2


def test_duplicate_page_is_removed(tmp_path: Path):
    path = tmp_path / "dup.avi"
    a, b = make_page(1), make_page(2)
    write_flip_video(path, [a, b, b, a.copy()])  # b でいったん動いて止まり直した、を再現
    out_base = tmp_path / "out"
    assert pagesnap.main(["--input", str(path), "--no-gemini", "--output-dir", str(out_base)]) == 0
    (out_dir,) = out_base.iterdir()
    assert len(list(out_dir.glob("page_*.jpg"))) == 2


def test_shaky_video_still_finds_each_page_once(tmp_path: Path):
    path = tmp_path / "shaky.avi"
    write_flip_video(path, [make_page(i) for i in range(12)], jitter=3, noise=6)
    out_base = tmp_path / "out"
    assert pagesnap.main(["--input", str(path), "--no-gemini", "--output-dir", str(out_base)]) == 0
    (out_dir,) = out_base.iterdir()
    assert len(list(out_dir.glob("page_*.jpg"))) == 12


def test_with_fake_gemini(flip_video: Path, tmp_path: Path, monkeypatch):
    calls = []

    class FakeModels:
        def generate_content(self, model, contents):
            calls.append(model)
            if len(calls) == 2:
                return SimpleNamespace(text=None, prompt_feedback=SimpleNamespace(block_reason="SAFETY"))
            return SimpleNamespace(text=f"### 解説 {len(calls)}\n根拠はこうです。")

    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(gemini, "make_client", lambda key: SimpleNamespace(models=FakeModels()))
    out_base = tmp_path / "out"
    code = pagesnap.main(["--input", str(flip_video), "--output-dir", str(out_base), "--model", "m"])
    assert code == 3  # 1ページだけ失敗 → 他は続行して終了コード 3
    (out_dir,) = out_base.iterdir()
    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert summary.count("根拠はこうです。") == 5
    assert "ブロックされました" in summary
    assert calls == ["m"] * 6


def test_missing_api_key(flip_video: Path, monkeypatch, capsys):
    for name in gemini.API_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert pagesnap.main(["--input", str(flip_video)]) == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_missing_input(tmp_path: Path):
    assert pagesnap.main(["--input", str(tmp_path / "none.mp4"), "--no-gemini"]) == 1

import numpy as np
import pytest

from pagesnap_lib.detection import (
    compute_dhash,
    dedupe_by_hash,
    hamming_distance,
    pick_representatives,
    preprocess_for_diff,
    segment_still_periods,
)
from tests.synthetic import make_page


def flat(value: int, shape=(40, 40)) -> np.ndarray:
    return np.full(shape, value, np.uint8)


class TestSegmentStillPeriods:
    def test_empty(self):
        assert segment_still_periods([]) == []

    def test_two_still_runs_separated_by_motion(self):
        frames = [flat(10)] * 4 + [flat(100), flat(200)] + [flat(50)] * 3
        assert segment_still_periods(frames, threshold=5, min_length=2) == [(0, 3), (6, 8)]

    def test_short_runs_are_dropped(self):
        frames = [flat(10)] * 3 + [flat(100)] + [flat(200)] * 2
        assert segment_still_periods(frames, threshold=5, min_length=3) == [(0, 2)]

    def test_whole_video_still(self):
        assert segment_still_periods([flat(10)] * 5, threshold=5, min_length=2) == [(0, 4)]

    def test_slow_drift_is_split(self):
        # 隣同士の差は 3 しかないが、先頭からは大きく変わっていく（ゆっくりめくった場合）
        frames = [flat(10 + 3 * i) for i in range(10)]
        periods = segment_still_periods(frames, threshold=5, min_length=2, drift_factor=2)
        assert len(periods) > 1
        assert segment_still_periods(frames, threshold=5, min_length=2, drift_factor=0) == [(0, 9)]

    def test_invalid_min_length(self):
        with pytest.raises(ValueError):
            segment_still_periods([flat(0)], min_length=0)


class TestPickRepresentatives:
    def test_middle(self):
        assert pick_representatives([(0, 4), (10, 11), (20, 20)]) == [2, 10, 20]

    def test_prefers_sharp_frame_in_center(self):
        sharp = [0.0] * 10
        sharp[0] = 999  # 端はブレや指が入りやすいので、いくら鋭くても選ばない
        sharp[6] = 50
        assert pick_representatives([(0, 9)], sharp) == [6]

    def test_ties_go_to_middle(self):
        assert pick_representatives([(0, 8)], [1.0] * 9) == [4]


class TestHash:
    def test_same_image_same_hash(self):
        page = make_page(1)
        assert hamming_distance(compute_dhash(page), compute_dhash(page.copy())) == 0

    def test_hash_length(self):
        assert compute_dhash(make_page(1), hash_size=8).shape == (64,)

    def test_noise_is_close_and_other_page_is_far(self):
        rng = np.random.default_rng(0)
        page = make_page(1)
        noisy = np.clip(page + rng.normal(0, 3, page.shape), 0, 255).astype(np.uint8)
        assert hamming_distance(compute_dhash(page), compute_dhash(noisy)) <= 20
        assert hamming_distance(compute_dhash(page), compute_dhash(make_page(2))) > 40

    def test_small_preprocessed_image_matches_full(self):
        page = make_page(3)
        d = hamming_distance(compute_dhash(page), compute_dhash(preprocess_for_diff(page)))
        assert d <= 20


class TestDedupe:
    def test_removes_duplicates_keeps_first(self):
        a, b, c = make_page(1), make_page(2), make_page(3)
        assert dedupe_by_hash([a, b, a.copy(), c, b.copy()]) == [0, 1, 3]

    def test_keeps_distinct_pages(self):
        pages = [make_page(i) for i in range(15)]
        assert dedupe_by_hash(pages) == list(range(15))

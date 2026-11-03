import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from component_eval import filter_frame, match_operating_point, sampled_frame_indices  # noqa: E402


def _roi():
    m = np.zeros((100, 100), bool)
    m[:, :50] = True
    return m


def test_sampler_matches_production_pattern():
    idx = sampled_frame_indices(25.0, 25)
    assert idx[:6] == [1, 4, 6, 9, 11, 14]
    assert len(idx) == 10


def test_filter_frame_cases():
    roi = _roi()
    # 1. in-ROI gt + overlapping in-ROI prediction: both kept
    ge, gi, pb, pi = filter_frame([[10, 10, 20, 20]], [1], [[11, 11, 20, 20]], [7], roi)
    assert len(ge) == 1 and len(pb) == 1
    # 2. gt outside ROI (bottom-center x=70): ignored, prediction on it dropped
    ge, gi, pb, pi = filter_frame([[60, 10, 20, 20]], [1], [[60, 10, 20, 20]], [7], roi)
    assert len(ge) == 0 and len(pb) == 0
    # 3. unmatched in-ROI prediction kept
    ge, gi, pb, pi = filter_frame(np.empty((0, 4)), [], [[10, 10, 20, 20]], [7], roi)
    assert len(pb) == 1
    # 4. prediction with bottom-center just outside, IoU>=0.5 to in-ROI gt: kept
    ge, gi, pb, pi = filter_frame([[30, 10, 20, 30]], [1], [[34, 10, 20, 30]], [7], roi)
    assert len(ge) == 1 and len(pb) == 1
    # 5. IoU .8 to ignore gt, <.5 to eval gt: dropped
    gt = [[0, 0, 40, 40], [60, 0, 40, 40]]  # eval (bc x=20), ignore (bc x=80)
    pr = [[62, 0, 40, 40]]  # IoU with ignore ~0.9; with eval 0
    ge, gi, pb, pi = filter_frame(gt, [1, 2], pr, [7], roi)
    assert len(ge) == 1 and len(pb) == 0


def test_match_operating_point():
    gt = [[0, 0, 10, 10]]
    # two predictions on one gt -> TP 1, FP 1
    r = match_operating_point([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], [1, 1], gt, [1])
    assert r[1] == (1, 1, 0)
    # right box wrong class -> FP for predicted class, FN for gt class
    r = match_operating_point([[0, 0, 10, 10]], [0.9], [2], gt, [1])
    assert r[2] == (0, 1, 0) and r[1] == (0, 0, 1)

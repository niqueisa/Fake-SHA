"""Tests for XLM-R confidence calibration spread."""

import torch

from inference.xlmr.analyzer import _calibrate_confidence


def test_calibrate_confidence_spreads_with_margin():
    decisive = torch.tensor([0.72, 0.28])
    borderline = torch.tensor([0.54, 0.46])

    conf_decisive = _calibrate_confidence(decisive, 0)
    conf_borderline = _calibrate_confidence(borderline, 0)

    assert conf_decisive > conf_borderline
    assert 0.52 <= conf_borderline <= 0.80
    assert 0.70 <= conf_decisive <= 0.97


def test_calibrate_confidence_stays_within_bounds():
    probs = torch.tensor([0.99, 0.01])
    conf = _calibrate_confidence(probs, 0)
    assert 0.52 <= conf <= 0.97

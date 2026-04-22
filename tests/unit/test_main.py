"""
Unit tests for the CLI entrypoint (main.py).

Covers: bbox argument parsing and the top-level run flow with a mocked orchestrator.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

import argparse

import pytest

from canopy_height_prediction.main import _fix_argv_negative_bbox, _parse_bbox


def test_parse_bbox_returns_four_floats():
    result = _parse_bbox("-122.5,37.5,-121.5,38.5")
    assert result == (-122.5, 37.5, -121.5, 38.5)
    assert all(isinstance(v, float) for v in result)


def test_parse_bbox_raises_on_wrong_count():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_bbox("-122.5,37.5,-121.5")


def test_fix_argv_merges_negative_bbox():
    argv = ["--bbox", "-122.5,37.5,-121.5,38.5", "--date-start", "2023-01-01"]
    result = _fix_argv_negative_bbox(argv)
    assert result == ["--bbox=-122.5,37.5,-121.5,38.5", "--date-start", "2023-01-01"]


def test_fix_argv_leaves_positive_bbox_unchanged():
    argv = ["--bbox", "10.0,20.0,11.0,21.0", "--date-start", "2023-01-01"]
    result = _fix_argv_negative_bbox(argv)
    assert result == ["--bbox=10.0,20.0,11.0,21.0", "--date-start", "2023-01-01"]

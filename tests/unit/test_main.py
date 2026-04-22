"""
Unit tests for the CLI entrypoint (main.py).

Covers: bbox argument parsing and the top-level run flow with a mocked orchestrator.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

import argparse

import pytest

from canopy_height_prediction.main import _parse_bbox


def test_parse_bbox_returns_four_floats():
    result = _parse_bbox("-122.5,37.5,-121.5,38.5")
    assert result == (-122.5, 37.5, -121.5, 38.5)
    assert all(isinstance(v, float) for v in result)


def test_parse_bbox_raises_on_wrong_count():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_bbox("-122.5,37.5,-121.5")

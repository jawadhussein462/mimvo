"""Tests for `Config` and the command-line parser."""

from __future__ import annotations

from mimvo.config import Config


def test_config_defaults():
    cfg = Config()
    assert cfg.fail_closed is True

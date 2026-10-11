"""Checks and scan sources found through entry points."""

from __future__ import annotations

import json
from importlib.metadata import EntryPoint

import pytest

import mimvo.plugins as plugins
from mimvo import Finding, MemoryRecord
from mimvo.checks.base import MemoryCheck
from mimvo.cli import EXIT_ERROR, EXIT_OK, main
from mimvo.exceptions import ConfigurationError, IntegrationError


class StaleCheck(MemoryCheck):
    name = "stale"
    needs_corpus = False

    def check(self, candidate, context):
        if "1999" in candidate.content:
            return [Finding(code="stale_fact", message="Mentions 1999.")]
        return []


class ListSource:
    def __init__(self, prefix: str = "r", count: str = "3"):
        self.prefix = prefix
        self.count = int(count)
        self.label = f"list:{prefix}"

    def records(self, *, batch_size=500, sample=None):
        n = self.count if sample is None else min(self.count, sample)
        for i in range(n):
            text = "party like it's 1999" if i == 1 else f"note {i}"
            yield MemoryRecord(id=f"{self.prefix}{i}", content=text)


NOT_A_CHECK = 42


def _points(monkeypatch, group_to_points):
    def fake(group):
        return {
            name: EntryPoint(name=name, value=value, group=group)
            for name, value in group_to_points.get(group, {}).items()
        }

    monkeypatch.setattr(plugins, "_entry_points", fake)


@pytest.fixture
def installed(monkeypatch):
    _points(
        monkeypatch,
        {
            plugins.CHECKS_GROUP: {
                "stale": "tests.unit.test_plugins:StaleCheck",
                "broken": "tests.unit.test_plugins_does_not_exist:Nope",
                "wrong": "tests.unit.test_plugins:NOT_A_CHECK",
            },
            plugins.SOURCES_GROUP: {"list": "tests.unit.test_plugins:ListSource"},
        },
    )


def test_listing(installed):
    assert plugins.available_checks() == ["broken", "stale", "wrong"]
    assert plugins.available_sources() == ["list"]


def test_load_check_builds_an_instance(installed):
    assert isinstance(plugins.load_check("stale"), StaleCheck)


def test_unknown_broken_and_wrong_plugins_are_clear_errors(installed):
    with pytest.raises(ConfigurationError, match="No check plugin named 'nope'"):
        plugins.load_check("nope")
    with pytest.raises(IntegrationError, match="failed to import"):
        plugins.load_check("broken")
    with pytest.raises(ConfigurationError, match="not callable"):
        plugins.load_check("wrong")


def test_load_source_passes_options(installed):
    source = plugins.load_source("list", {"prefix": "x", "count": "2"})
    assert [r.id for r in source.records()] == ["x0", "x1"]
    with pytest.raises(ConfigurationError, match="source plugin 'list'"):
        plugins.load_source("list", {"no_such_option": "1"})


def test_cli_scans_a_plugin_source_with_a_plugin_check(installed, tmp_path, capsys):
    out = tmp_path / "report.json"
    code = main(
        [
            "scan",
            "plugin",
            "list",
            "--option",
            "prefix=p",
            "--option",
            "count=4",
            "--check",
            "stale",
            "--json",
            str(out),
        ]
    )
    assert code == EXIT_OK
    data = json.loads(out.read_text())
    assert data["source"] == "list:p" and data["total"] == 4
    assert [(f["record_id"], f["code"]) for f in data["findings"]] == [("p1", "stale_fact")]
    assert "stale" in data["checks"]


def test_cli_reports_a_missing_plugin(installed, capsys):
    assert main(["scan", "plugin", "nope"]) == EXIT_ERROR
    assert "No source plugin named 'nope'" in capsys.readouterr().err


def test_cli_lists_plugins(installed, capsys):
    assert main(["plugins"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "  stale" in out and "  list" in out


def test_cli_rejects_a_malformed_option(installed):
    with pytest.raises(SystemExit):
        main(["scan", "plugin", "list", "--option", "nokey"])

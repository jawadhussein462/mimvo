"""Measure the default Mimvo checks on public datasets.

    python benchmarks/run.py                      # held-out test split
    python benchmarks/run.py --split dev --errors 20
    python benchmarks/run.py --write benchmarks/RESULTS.md

Each sample is scanned on its own, as one record, with `Mimvo()`
defaults. A benign sample counts as a false positive when any finding is
raised. An attack counts as caught when any finding is raised; the
"targeted check" column narrows that to the check the dataset is about
(injection for injection sets, poisoning for poisoning sets).

`--errors N` prints misclassified samples, and only for the `dev` split:
the `test` split is for scoring, not for tuning patterns.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.datasets import (  # noqa: E402
    BENIGN,
    INJECTION,
    POISONING,
    SOURCES,
    Sample,
    load_all,
)
from mimvo import Mimvo, __version__  # noqa: E402
from mimvo.models.enums import Severity  # noqa: E402

TARGET_CHECK = {INJECTION: "injection", POISONING: "poisoning"}


def _evaluate(samples: Iterable[Sample]) -> list[tuple[Sample, set[str], Severity | None]]:
    guard = Mimvo()
    out = []
    for i, sample in enumerate(samples):
        report = guard.scan([{"id": f"s{i}", "content": sample.text}])
        checks = {f.check for f in report.findings}
        out.append((sample, checks, report.worst_severity()))
    return out


def _pct(n: int, d: int) -> str:
    return "—" if d == 0 else f"{100 * n / d:.1f}%"


def render(results: list[tuple[Sample, set[str], Severity | None]], split: str) -> str:
    """Build the Markdown results tables.

    Args:
        results: `(sample, checks that raised a finding, worst severity)` per sample.
        split: Which split was scored, for the heading.

    Returns:
        Markdown text.
    """
    by_dataset: dict[str, list[tuple[Sample, set[str], Severity | None]]] = defaultdict(list)
    for row in results:
        by_dataset[row[0].dataset].append(row)

    lines = [
        f"### Split: `{split}` · Mimvo {__version__} · default checks",
        "",
        "**Benign memories** (lower is better)",
        "",
        "| Dataset | Samples | Flagged (any finding) | Flagged at high or above |",
        "| :-- | --: | --: | --: |",
    ]
    totals = [0, 0, 0]
    for name, rows in sorted(by_dataset.items()):
        if rows[0][0].label != BENIGN:
            continue
        flagged = sum(1 for _, checks, _ in rows if checks)
        high = sum(1 for _, _, sev in rows if sev is not None and sev.is_at_least(Severity.HIGH))
        totals = [totals[0] + len(rows), totals[1] + flagged, totals[2] + high]
        lines.append(
            f"| {name} | {len(rows):,} | {flagged:,} ({_pct(flagged, len(rows))}) "
            f"| {high:,} ({_pct(high, len(rows))}) |"
        )
    lines.append(
        f"| **All benign** | **{totals[0]:,}** | **{totals[1]:,} ({_pct(totals[1], totals[0])})** "
        f"| **{totals[2]:,} ({_pct(totals[2], totals[0])})** |"
    )
    lines += [
        "",
        "**Attacks** (higher is better)",
        "",
        "| Dataset | Samples | Caught (any finding) | Caught at high or above"
        " | By targeted check |",
        "| :-- | --: | --: | --: | --: |",
    ]
    for name, rows in sorted(by_dataset.items()):
        label = rows[0][0].label
        if label == BENIGN:
            continue
        target = TARGET_CHECK[label]
        caught = [row for row in rows if row[1]]
        targeted = sum(1 for row in rows if target in row[1])
        high = sum(1 for _, _, sev in caught if sev is not None and sev.is_at_least(Severity.HIGH))
        lines.append(
            f"| {name} | {len(rows):,} | {len(caught):,} ({_pct(len(caught), len(rows))})"
            f" | {high:,} ({_pct(high, len(rows))}) | {target}: {targeted:,}"
            f" ({_pct(targeted, len(rows))}) |"
        )
    return "\n".join(lines) + "\n"


def corpus_mode(samples: list[Sample]) -> str:
    """Scan poisoned passages and benign memories together, as one store.

    Per-record scanning cannot see multi-document poisoning: PoisonedRAG
    plants five paraphrases per target. Here every PoisonedRAG passage and
    every benign sample go into one batch, so `TrustRAGDetector` can find
    the clusters. The detector's thresholds were not changed in this work,
    so the whole dataset is used, not a split.

    Returns:
        A Markdown table.
    """
    batch = [s for s in samples if s.label in {POISONING, BENIGN}]
    started = time.perf_counter()
    report = Mimvo().scan({"id": f"s{i}", "content": s.text} for i, s in enumerate(batch))
    elapsed = time.perf_counter() - started
    clustered = {f.record_id for f in report.findings if f.code == "poisoning_cluster"}
    poison = [f"s{i}" for i, s in enumerate(batch) if s.label == POISONING]
    benign = [f"s{i}" for i, s in enumerate(batch) if s.label == BENIGN]
    hit = sum(1 for rid in poison if rid in clustered)
    false = sum(1 for rid in benign if rid in clustered)
    return "\n".join(
        [
            "### Corpus mode: PoisonedRAG passages and benign memories in one store",
            "",
            "| | Records | Flagged as `poisoning_cluster` |",
            "| :-- | --: | --: |",
            f"| PoisonedRAG passages | {len(poison):,} | {hit:,} ({_pct(hit, len(poison))}) |",
            f"| Benign memories | {len(benign):,} | {false:,} ({_pct(false, len(benign))}) |",
            "",
            f"One scan of {report.total:,} records, no stored vectors (lexical fallback), "
            f"{elapsed:.1f} s.",
            "",
        ]
    )


def _errors(results: list[tuple[Sample, set[str], Severity | None]], limit: int) -> str:
    fps = [s for s, checks, _ in results if s.label == BENIGN and checks]
    fns = [s for s, checks, _ in results if s.label != BENIGN and not checks]
    out = [f"\n-- false positives ({len(fps)}) --"]
    out += [f"[{s.dataset}] {s.text[:200]!r}" for s in fps[:limit]]
    out.append(f"\n-- misses ({len(fns)}) --")
    out += [f"[{s.dataset}] {s.text[:200]!r}" for s in fns[:limit]]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print (or write) the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, default=ROOT / ".benchmark-cache")
    parser.add_argument("--split", choices=["test", "dev", "all"], default="test")
    parser.add_argument("--errors", type=int, default=0, metavar="N")
    parser.add_argument("--write", type=Path, default=None, metavar="PATH")
    args = parser.parse_args(argv)
    if args.errors and args.split != "dev":
        parser.error("--errors is only available for --split dev")

    samples = load_all(args.cache)
    if args.split != "all":
        samples = [s for s in samples if s.split == args.split]
    started = time.perf_counter()
    results = _evaluate(samples)
    elapsed = time.perf_counter() - started

    text = render(results, args.split)
    text += (
        f"\n{len(results):,} samples scanned in {elapsed:.1f} s. Sources (pinned commits): "
        + ", ".join(f"[{s.name}]({s.url}/tree/{s.commit}) ({s.licence})" for s in SOURCES.values())
        + ".\n\n"
    )
    text += corpus_mode(load_all(args.cache))
    print(text)
    if args.errors:
        print(_errors(results, args.errors))
    if args.write:
        header = (
            "# Benchmark results\n\n"
            f"Generated by `python benchmarks/run.py --split {args.split} --write {args.write}`. "
            "See [README.md](README.md) for the method.\n\n"
        )
        args.write.write_text(header + text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Contributing to Mimvo

Thanks for your interest in improving Mimvo. This project aims to be a
familiar, well-typed, batteries-included toolkit — contributions should preserve
that feel.

## Development setup

Mimvo requires Python 3.11+. We recommend [uv](https://docs.astral.sh/uv/).

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

## The toolchain

```bash
ruff check mimvo tests examples      # lint
ruff format mimvo tests examples     # format
mypy mimvo                           # type-check
pytest                                  # tests
```

All four must pass. CI runs them on Python 3.11–3.13.

## Design principles

- **CLI is the front door.** Most users run `mimvo scan`.
- **Find, then fix.** Scan finds poisoned facts, hidden instructions, and
  leaked secrets. The HTML report explains the recommended action.
- **Deterministic first.** Checks are cheap, offline, and fully functional
  with zero external services.
- **Extension via Protocols, not inheritance.** New checks and scan sources
  should satisfy the relevant `typing.Protocol`.
- **Never log or store secrets.** Findings, reports, and telemetry must not
  include secret values. Scan snippets are masked.
- **Scan connections are read-only.** Scan sources list and fetch; they never
  upsert or delete.
- **Sync is always sync; async is always async.** No method returns a coroutine
  conditionally.

## Adding a detector (a new method for an existing security concern)

Security checks are organised as one folder per concern, one module per
method: `mimvo/checks/security/{injection,poisoning,secrets}/<method>.py`.

1. Subclass `BaseDetector` (or `HFTextClassifierDetector` /
   `HFTokenClassifierDetector` from `security/_hf.py` for Hugging Face
   models) and set a unique `name`.
2. Implement `detect_text(text)` — or `detect(candidate, context)` if you need
   the query or neighbouring records — and return `Detection`s via
   `self.hit(...)`. Use `code=` to pick one of the parent check's `rules`,
   and pass `score=` (0 to 1) so findings get a confidence.
   A detector that overrides `detect` must declare `needs_corpus`: `True`
   runs it after the whole store is read, `False` runs it while records
   stream (use that when it only reads the candidate). Neighbour
   searches go through `mimvo.corpus.corpus_of(context)`, never a loop
   over `context.existing`.
3. **Never** put matched text or secret values in evidence: kinds, labels,
   counts, and scores only.
4. Never raise to signal a finding, and never turn an internal error into a
   finding: an exception is recorded as a scan error and the scan is marked
   incomplete.
5. Import optional dependencies lazily inside `_load()`, raise
   `ConfigurationError` with the install hint, and accept an injectable
   inference callable (`classify=`, `tag=`, `transport=`, ...) so the detector
   is testable offline. Add an extra to `pyproject.toml` and a mypy override.
6. Export it from the folder's `__init__.py`, add a row to the README table,
   and add fake-driven tests under `tests/unit/`.
7. For a change to a default detector, run `python benchmarks/run.py` before
   and after. Tune on `--split dev` only, and put the new `test` numbers in
   the PR.

## Adding a check

1. Add the check's rules to the catalogue in `mimvo/rules.py`, one
   `_rule(...)` per finding code, with the check's name as the category.
   This is the only place severity, action, message, and remediation live.
   Set the action to `review`, `quarantine`, or `delete`. Pick severities by
   the rule in the README: the text is the attack (high), model or
   behavioural evidence (medium), a statistical pattern across the store
   (low).
2. Subclass `SecurityCheck` in a new
   `mimvo/checks/security/<concern>/` folder with `name`, `default_code`,
   `rules = rules_in("<concern>")`, `default_detectors()`, and at least a
   heuristic detector. (A check outside this repo lists its own `Rule`s
   with `rule_map([...])`; defining the class registers them.)
3. Add it to `default_checks()` only if it should run by default (defaults
   must stay offline and dependency-free).
4. Add unit tests under `tests/unit/` and, if security-relevant, an invariant
   test under `tests/security/`.

## Adding a scan source

1. Add `mimvo/sources/<store>.py` implementing `records(batch_size=..., sample=...)`
   as a read-only iterator; the helpers in `mimvo/sources/base.py` turn store
   rows into `MemoryRecord`s. Output formats live in `mimvo/reporters/`.
2. Stream in batches. Honour `--sample`. Never write.
3. Import the provider SDK lazily, or recognise the client by its attributes
   without importing it; add an optional extra in `pyproject.toml`.
4. Return stored vectors when the store has them (the vector detectors need
   them), and keep the store's own ids.
5. Wire it into `mimvo scan <name>` in `mimvo/cli.py`.
6. Test against an in-process fake in `tests/unit/`, and against the real
   client in `tests/integration/test_backends.py` (skipped when the client is
   not installed; the CI `backends` job installs it).

A store or check that does not belong in this repository can live in its
own package and register under the `mimvo.sources` or `mimvo.checks`
entry-point group; see "Packaging checks and sources as plugins" in the
README.

## Commit / PR expectations

- Keep the public API small and documented (docstrings on public APIs).
- Update `CHANGELOG.md` under `[Unreleased]`.
- Security-relevant changes must include tests that would fail without the fix.

## Reporting security issues

Please see [SECURITY.md](SECURITY.md) — do not open public issues for
vulnerabilities.

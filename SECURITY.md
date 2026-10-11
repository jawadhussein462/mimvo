# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately rather than opening a public
issue. Email the maintainers (see repository metadata) with a description, steps
to reproduce, and impact. We aim to acknowledge reports promptly and to
coordinate disclosure.

Do **not** include real secrets, credentials, or production customer data in
reports.

## Scope and threat model

Mimvo is a **trust boundary** between an AI agent and a long-term
knowledge memory (RAG store, user memory, retrieved documents). It is
designed to reduce the risk that persistent memory becomes poisoned, injected,
or secret-bearing — [OWASP ASI06: Memory & Context Poisoning](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications/).
It is one layer in a defense-in-depth strategy — not a complete security solution.

The product is oriented around three security checks:

- **Poisoning** — false or attacker-controlled facts.
- **Injection** — hidden / persistent instructions.
- **Secrets** — credentials and secret-bearing content.

### What Mimvo enforces reliably (deterministic)

- **Secret values** are never included in findings, scan reports, or telemetry.
  Secret findings recommend **delete**.
- **Critical findings** (poisoning, injection) are reported with a recommended
  action of review, quarantine, or delete.

These are covered by explicit invariant tests.

## Important limitations — please read

The **default** detectors are **heuristic** and are **not complete or
production-perfect**. They produce strong *signals*, not guarantees.

- **Poisoning detection** is pattern-based. It looks for false
  security-relevant facts (disabled auth, attacker hosts, approval bypasses).
  It will miss novel phrasings, obfuscation, encodings, and non-English text.
- **Injection / persistent-instruction detection** is pattern-based. It will
  miss novel phrasings, obfuscation, encodings, and non-English text, and can
  produce false positives.
- **Secret detection** matches common, high-signal credential formats (API
  keys, tokens, private keys, credential assignments). It will miss custom or
  unusual secret formats. Do not rely on it as your only secret scanner.

Each security check accepts additional **detectors** (classifier models,
hosted APIs, statistical filters; see `mimvo.checks.security`). Notes on
those:

- Model detectors are only as good as their training distribution; every
  published injection classifier shows measurable false positives on
  security-adjacent benign text. Calibrate `threshold` on your own data and
  consider `min_detectors=2` when stacking noisy methods.
- Hosted detectors (Azure Prompt Shields, Lakera Guard) send the candidate
  text to a third party. Only the text is sent; nothing is logged locally.
- A detector that raises (model not downloaded, network error, bad API key)
  is listed in `ScanReport.errors` and the scan is marked incomplete; the
  other detectors' results are kept. A failure is never reported as a
  finding, so it can never produce a "delete" recommendation that automation
  could act on. Under `fail_closed=True` the CLI exits `2` and `WriteGuard` /
  `RetrieveGuard` treat the affected records as unsafe.
- Detector evidence carries kinds, labels, and scores — never the matched
  text or secret values. This is an invariant for built-in detectors and a
  requirement for custom ones. Scan reports include only masked snippets.

## Operational guidance

- Keep `fail_closed=True` (default) in production so an incomplete scan fails
  the pipeline instead of passing as clean.
- If you automate cleanup from `ScanReport.action_plan()`, gate it on
  `report.complete` and a confidence floor, and quarantine before deleting.
- Stack detectors on the built-in checks when the heuristics are not enough.

## Supported versions

During the 0.x series, only the latest released version receives security fixes.

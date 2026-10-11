"""Example 7: scan for leaked secrets.

A JSONL export of agent memory is screened on this machine. Secret values
do not appear in the report.

Run it with `python examples/07_secret_detection.py`.

The same scan from the command line:

    mimvo scan jsonl export.jsonl --report report.html
"""

from __future__ import annotations

from mimvo import MemoryRecord, Mimvo


def main() -> None:
    records = [
        MemoryRecord(id="ok", content="Alice prefers annual billing."),
        MemoryRecord(id="leak", content="The billing API key is sk-abcdefghijklmnop1234567890."),
    ]
    report = Mimvo().scan(records)
    print(report)
    for finding in report.findings:
        print(finding.record_id, finding.code, finding.recommended_action.value, finding.snippet)


if __name__ == "__main__":
    main()

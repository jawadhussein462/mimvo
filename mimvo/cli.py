"""Command line: `mimvo scan`.

Scan an existing store for poisoned facts, hidden instructions, and leaked
secrets. The HTML report says what to do with each hit; JSON, SARIF, and
Markdown carry the same findings to tickets, code scanning, and CI pages.
Connections are read-only: the command lists and fetches, and it does not
write to the store.

Exit codes follow other security scanners: `0` when the scan finished,
`1` when `--fail-on` is set and a finding reached that severity, `2` on a
usage, connection, or file error, or when a check or detector failed so
the scan is incomplete (unless `--allow-incomplete`).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .client import Mimvo
from .exceptions import ConfigurationError, IntegrationError, MimvoError
from .models.enums import Severity
from .models.memory import MemoryRecord
from .models.results import ScanReport
from .plugins import available_checks, available_sources, load_check, load_source
from .reporters import format_scan_summary, render_html, render_markdown, render_sarif
from .sources.base import DEFAULT_BATCH_SIZE, ScanSource
from .sources.chroma import ChromaScanSource
from .sources.jsonl import JsonlScanSource
from .sources.langchain import LangChainScanSource
from .sources.langgraph import LangGraphStoreScanSource
from .sources.mem0 import Mem0ScanSource
from .sources.pgvector import PgVectorScanSource
from .sources.pinecone import PineconeScanSource
from .sources.qdrant import QdrantScanSource

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2

_SEVERITY_CHOICES = [s.value for s in Severity]

_DESCRIPTION = (
    "Scan your AI agent's memory for poisoned facts, hidden instructions "
    "and leaked secrets, locally, in minutes."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mimvo", description=_DESCRIPTION)
    parser.add_argument("--version", action="version", version=f"mimvo {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser(
        "scan",
        help="Find poisoned facts, hidden instructions, and leaked secrets.",
        description=(
            "Find problems in an existing memory store. Connections are "
            "read-only. Use --report for an HTML file you can forward, "
            "--json for machine-readable findings, --sarif for GitHub code "
            "scanning, --markdown for a CI job summary, --fail-on to gate a "
            "pipeline, and --sample to cap very large stores."
        ),
    )
    sources = scan.add_subparsers(dest="source", required=True)

    chroma = sources.add_parser("chroma", help="Scan a Chroma persistent collection.")
    chroma.add_argument("--path", required=True, help="Path to the Chroma database directory.")
    chroma.add_argument("--collection", required=True, help="Collection name.")
    _add_scan_output_flags(chroma)

    qdrant = sources.add_parser("qdrant", help="Scan a Qdrant collection.")
    qdrant.add_argument("--url", required=True, help="Qdrant HTTP URL.")
    qdrant.add_argument("--collection", required=True, help="Collection name.")
    qdrant.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY"))
    qdrant.add_argument("--text-field", default=None, help="Payload field that holds the text.")
    _add_scan_output_flags(qdrant)

    pgvector = sources.add_parser("pgvector", help="Scan a Postgres / pgvector table.")
    pgvector.add_argument("--dsn", required=True, help="Postgres connection string.")
    pgvector.add_argument("--table", required=True, help="Table name (schema.table allowed).")
    pgvector.add_argument("--text-column", required=True, help="Column that holds memory text.")
    pgvector.add_argument("--id-column", default="id", help="Column that holds the record id.")
    pgvector.add_argument(
        "--embedding-column",
        default=None,
        help="Column that holds the stored vector (enables hubness and TrustRAG NN).",
    )
    pgvector.add_argument(
        "--created-at-column",
        default=None,
        help="Column that holds the insert time (enables temporal NLI).",
    )
    _add_scan_output_flags(pgvector)

    pinecone = sources.add_parser("pinecone", help="Scan a Pinecone index.")
    pinecone.add_argument("--index", required=True, help="Index name.")
    pinecone.add_argument("--api-key", default=os.environ.get("PINECONE_API_KEY"))
    pinecone.add_argument("--host", default=None, help="Index host (serverless).")
    pinecone.add_argument("--namespace", default="")
    pinecone.add_argument(
        "--text-field",
        default=None,
        help="Metadata field that holds the text (default: content/text/document).",
    )
    _add_scan_output_flags(pinecone)

    jsonl = sources.add_parser("jsonl", help="Scan a JSON Lines export.")
    jsonl.add_argument("path", help="JSON Lines file, or '-' for stdin.")
    _add_scan_output_flags(jsonl)

    langchain = sources.add_parser(
        "langchain",
        help="Scan a LangChain vector store or a LangGraph long-term memory store.",
        description=(
            "Your code builds the store; --factory names it as module:attribute (a store, or a "
            "function that returns one), imported from the current directory."
        ),
    )
    langchain.add_argument(
        "--factory",
        required=True,
        metavar="MODULE:ATTR",
        help="Where to get the store, e.g. myapp.memory:get_vector_store.",
    )
    langchain.add_argument(
        "--namespace",
        default=None,
        help="LangGraph stores: namespace prefix to read, slash-separated (e.g. memories/alice).",
    )
    langchain.add_argument(
        "--text-field",
        default=None,
        help="LangGraph stores: key in each value that holds the memory text.",
    )
    _add_scan_output_flags(langchain)

    mem0 = sources.add_parser(
        "mem0",
        help="Scan mem0: open source with --config, or the hosted platform with --api-key.",
    )
    target = mem0.add_mutually_exclusive_group()
    target.add_argument(
        "--config",
        metavar="PATH",
        help="mem0 config (JSON or YAML) for Memory.from_config, the same file your app uses.",
    )
    target.add_argument(
        "--api-key",
        default=None,
        help="mem0 platform API key (or MEM0_API_KEY). Needs --user-id, --agent-id, or --run-id.",
    )
    mem0.add_argument("--user-id", default=None)
    mem0.add_argument("--agent-id", default=None)
    mem0.add_argument("--run-id", default=None)
    _add_scan_output_flags(mem0)

    plugin = sources.add_parser(
        "plugin",
        help="Scan a store through a source another package provides (see `mimvo plugins`).",
        description=(
            "Read a store through a scan source registered under the 'mimvo.sources' "
            "entry-point group. --option key=value is passed to its constructor."
        ),
    )
    plugin.add_argument("name", help="The source plugin's name.")
    plugin.add_argument(
        "--option",
        dest="options",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        type=_option,
        help="Constructor argument for the source. Repeat for several.",
    )
    _add_scan_output_flags(plugin)

    sub.add_parser(
        "plugins",
        help="List the checks and scan sources installed packages provide.",
    )
    return parser


def _option(value: str) -> tuple[str, str]:
    key, sep, val = value.partition("=")
    if not sep or not key.strip():
        raise argparse.ArgumentTypeError(f"{value!r} is not KEY=VALUE")
    return key.strip(), val


def _add_scan_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--report",
        metavar="PATH",
        help="Write a single HTML report (forward this to your boss).",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        metavar="PATH",
        help="Write the full report as JSON (versioned schema, one object per finding).",
    )
    parser.add_argument(
        "--sarif",
        dest="sarif_path",
        metavar="PATH",
        help="Write SARIF 2.1.0 for GitHub code scanning. Snippets are left out.",
    )
    parser.add_argument(
        "--markdown",
        dest="markdown_path",
        metavar="PATH",
        help="Write a Markdown summary, e.g. to $GITHUB_STEP_SUMMARY or a PR comment.",
    )
    parser.add_argument(
        "--fail-on",
        choices=_SEVERITY_CHOICES,
        metavar="SEVERITY",
        default=None,
        help=(
            "Exit 1 when any finding is at this severity or worse "
            f"({', '.join(_SEVERITY_CHOICES)}). Default: never fail on findings."
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=_confidence,
        metavar="SCORE",
        default=None,
        help=(
            "With --fail-on, ignore findings whose confidence is below SCORE (0 to 1). "
            "Unscored findings still count."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Exit 0 even when a check or detector failed (missing model, bad API key). "
            "By default a failure makes the scan incomplete and exits 2."
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Print only the counts, not the table of findings.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        default=None,
        help="Scan at most N records (for very large stores).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Records fetched per round-trip (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--check",
        dest="plugin_checks",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Also run a check another package provides (see `mimvo plugins`). A plugin "
            "check named like a default one replaces it. Repeat for several."
        ),
    )


def _confidence(value: str) -> float:
    try:
        score = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if not 0.0 <= score <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return score


def _records_for(args: argparse.Namespace) -> tuple[Iterator[MemoryRecord], str]:
    sample = args.sample
    batch = args.batch_size
    source = args.source
    src: ScanSource
    if source == "chroma":
        src = ChromaScanSource(path=args.path, collection=args.collection)
        return src.records(batch_size=batch, sample=sample), f"chroma:{args.collection}"
    if source == "qdrant":
        src = QdrantScanSource(
            url=args.url,
            collection=args.collection,
            api_key=args.api_key,
            text_field=args.text_field,
        )
        return src.records(batch_size=min(batch, 256), sample=sample), f"qdrant:{args.collection}"
    if source == "pgvector":
        src = PgVectorScanSource(
            dsn=args.dsn,
            table=args.table,
            text_column=args.text_column,
            id_column=args.id_column,
            embedding_column=args.embedding_column,
            created_at_column=args.created_at_column,
        )
        return src.records(batch_size=batch, sample=sample), f"pgvector:{args.table}"
    if source == "pinecone":
        src = PineconeScanSource(
            index=args.index,
            api_key=args.api_key,
            host=args.host,
            namespace=args.namespace,
            text_field=args.text_field,
        )
        return src.records(batch_size=min(batch, 100), sample=sample), f"pinecone:{args.index}"
    if source == "langchain":
        store = _load_factory(args.factory)
        if callable(getattr(store, "similarity_search", None)) or not callable(
            getattr(store, "list_namespaces", None)
        ):
            vector_source = LangChainScanSource(store)
            return vector_source.records(batch_size=batch, sample=sample), vector_source.label
        namespace = tuple(part for part in (args.namespace or "").split("/") if part)
        graph_source = LangGraphStoreScanSource(
            store, namespace=namespace, text_field=args.text_field
        )
        return graph_source.records(batch_size=batch, sample=sample), graph_source.label
    if source == "mem0":
        mem0_source = Mem0ScanSource(
            _mem0_client(args.config, args.api_key),
            user_id=args.user_id,
            agent_id=args.agent_id,
            run_id=args.run_id,
        )
        return mem0_source.records(batch_size=batch, sample=sample), mem0_source.label
    if source == "jsonl":
        if args.path == "-":
            src = JsonlScanSource(sys.stdin)
            label = "jsonl:stdin"
        else:
            src = JsonlScanSource(args.path)
            label = f"jsonl:{args.path}"
        return src.records(sample=sample), label
    if source == "plugin":
        src = load_source(args.name, dict(args.options))
        return src.records(batch_size=batch, sample=sample), str(
            getattr(src, "label", None) or f"plugin:{args.name}"
        )
    raise ConfigurationError(f"unknown scan source {source!r}")


def _load_factory(spec: str) -> Any:
    """Import `module:attr` from the current directory and return the store.

    A class or a function is called with no arguments and must return the
    store; a store instance is used as it is.
    """
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ConfigurationError(f"--factory must look like module:attribute, got {spec!r}")
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigurationError(f"cannot import {module_name!r}: {exc}") from exc
    target: Any = module
    for part in attr.split("."):
        if not hasattr(target, part):
            raise ConfigurationError(f"{module_name!r} has no attribute {attr!r}")
        target = getattr(target, part)
    is_store = callable(getattr(target, "similarity_search", None)) or callable(
        getattr(target, "search", None)
    )
    if isinstance(target, type) or (callable(target) and not is_store):
        target = target()
    return target


def _mem0_client(config_path: str | None, api_key: str | None) -> Any:
    """Build `Memory.from_config(config)` or `MemoryClient(api_key=...)`."""
    try:
        import mem0
    except ImportError as exc:
        raise IntegrationError('mem0 support requires `pip install "mimvo[mem0]"`.') from exc
    if config_path:
        text = Path(config_path).read_text(encoding="utf-8")
        if config_path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc:
                raise IntegrationError("reading a YAML config needs PyYAML.") from exc
            config = yaml.safe_load(text)
        else:
            config = json.loads(text)
        if not isinstance(config, dict):
            raise ConfigurationError(f"{config_path}: expected a mapping")
        return mem0.Memory.from_config(config)
    key = api_key or os.environ.get("MEM0_API_KEY")
    if not key:
        raise ConfigurationError(
            "mem0 scan needs --config (open source) or --api-key / MEM0_API_KEY."
        )
    return mem0.MemoryClient(api_key=key)


def _cmd_scan(args: argparse.Namespace) -> int:
    try:
        guard = Mimvo(
            fail_closed=not args.allow_incomplete,
            checks=[load_check(name) for name in args.plugin_checks],
        )
        records, label = _records_for(args)
        report = guard.scan(records)
    except (OSError, MimvoError, ValueError) as exc:
        print(f"mimvo scan: {exc}", file=sys.stderr)
        return EXIT_ERROR
    report.source = label
    report.sample = args.sample

    report_path: str | None = None
    json_path: str | None = None
    outputs: list[tuple[str, str]] = []
    writers: list[tuple[str | None, str, Callable[[ScanReport], str]]] = [
        (args.report, "html", render_html),
        (args.json_path, "json", lambda r: r.model_dump_json(indent=2) + "\n"),
        (args.sarif_path, "SARIF", render_sarif),
        (args.markdown_path, "Markdown summary", render_markdown),
    ]
    try:
        for path, kind, render in writers:
            if not path:
                continue
            Path(path).write_text(render(report), encoding="utf-8")
            if kind == "html":
                report_path = path
            elif kind == "json":
                json_path = path
            else:
                outputs.append((kind, path))
    except OSError as exc:
        print(
            _summary(report, args, report_path=report_path, json_path=json_path, outputs=outputs),
            flush=True,
        )
        print(f"mimvo scan: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(
        _summary(report, args, report_path=report_path, json_path=json_path, outputs=outputs),
        flush=True,
    )
    if not report.complete and guard.config.fail_closed:
        print(
            f"mimvo scan: incomplete: {report.records_with_errors:,} records were not fully "
            "checked because a check or detector failed (see above). Fix it and scan again, "
            "or pass --allow-incomplete.",
            file=sys.stderr,
        )
        return EXIT_ERROR
    if args.fail_on:
        threshold = Severity(args.fail_on)
        failing = report.at_or_above(threshold, min_confidence=args.min_confidence)
        if failing:
            noun = "finding" if len(failing) == 1 else "findings"
            gate = f"--fail-on {threshold.value}"
            if args.min_confidence is not None:
                gate += f" --min-confidence {args.min_confidence:g}"
            print(
                f"mimvo scan: {len(failing):,} {noun} at {threshold.value} or above ({gate})",
                file=sys.stderr,
            )
            return EXIT_FINDINGS
    return EXIT_OK


def _cmd_plugins() -> int:
    checks, sources = available_checks(), available_sources()
    print("Checks (mimvo scan ... --check NAME):")
    print("".join(f"  {name}\n" for name in checks).rstrip() or "  none installed")
    print("Sources (mimvo scan plugin NAME --option KEY=VALUE):")
    print("".join(f"  {name}\n" for name in sources).rstrip() or "  none installed")
    return EXIT_OK


def _summary(
    report: ScanReport,
    args: argparse.Namespace,
    *,
    report_path: str | None = None,
    json_path: str | None = None,
    outputs: list[tuple[str, str]] | None = None,
) -> str:
    return format_scan_summary(
        report,
        report_path=report_path,
        json_path=json_path,
        color=_color_enabled(),
        details=not args.quiet,
        outputs=outputs or (),
    )


def _color_enabled() -> bool:
    """Decide whether the scan summary may use terminal color.

    Returns:
        `False` when `NO_COLOR` is set to any non-empty value, or when
        `TERM` is `dumb`. Otherwise `True`.

        A missing TTY does not turn color off. IDE runners often fail
        `isatty` and would otherwise print the summary in plain text.
    """
    if os.environ.get("NO_COLOR", "") != "":
        return False
    return os.environ.get("TERM") != "dumb"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line tool.

    Args:
        argv: Arguments after the program name, such as
            `["scan", "jsonl", "export.jsonl"]`. `None` reads `sys.argv`.

    Returns:
        `0` when the scan finished and any requested files were written.
        `1` when `--fail-on` is set and a finding reached that severity.
        `2` when the arguments, the store, or a file write failed, or when
        the scan is incomplete and `--allow-incomplete` was not passed.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "scan":
        return _cmd_scan(args)
    if args.command == "plugins":
        return _cmd_plugins()
    parser.error("unknown command")  # pragma: no cover
    return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

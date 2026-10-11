"""Settings for one `Mimvo` instance.

Pass a `Config` into the constructor. Nothing here is read from a global.
Every field has a default, so `Mimvo()` works with no arguments.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Config(BaseModel):
    """Settings a check can read while it runs.

    Attributes:
        fail_closed: What an incomplete scan means. A check or detector that
            raises (missing model, network error, bad API key) is always
            logged and listed in `ScanReport.errors`, and never becomes a
            finding. With `True` (the default) that incompleteness counts as
            a failure: the CLI exits `2`, `WriteGuard` refuses the write, and
            `RetrieveGuard` drops the record. With `False` the parts that ran
            are trusted on their own.
        batch_size: How many records each check is handed at once, per
            pass. Detectors that batch (Hugging Face models, hosted APIs
            with `concurrency`) do one call per batch. Larger batches are
            faster on a GPU and hold more records in memory at a time.
    """

    model_config = ConfigDict(extra="forbid")

    fail_closed: bool = Field(
        default=True,
        description=(
            "Treat an incomplete scan (a check or detector raised) as a failure: "
            "the CLI exits 2 and the guards block the affected records."
        ),
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Records handed to each check at once, per pass.",
    )


__all__ = ["Config"]

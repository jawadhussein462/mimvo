"""Shared bases for Hugging Face `transformers` detectors.

`transformers` and `torch` are imported on first use, so the core package
stays installable without them. Every detector accepts a callable
(`classify=` or `tag=`) so tests can fake the model and so you can point
the detector at a remote endpoint without loading weights locally.

Install the models with `pip install "mimvo[hf]"`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from ...context import CheckContext
from ...exceptions import ConfigurationError, IntegrationError
from ...models.memory import MemoryCandidate
from .base import BaseDetector, Detection

# One label prediction: ``{"label": "MALICIOUS", "score": 0.99}``.
LabelScores = list[dict[str, Any]]
Classify = Callable[[str], LabelScores]
ClassifyBatch = Callable[[list[str]], list[LabelScores]]
# One tagged entity: ``{"entity_group": "PASSWORD", "score": 0.98, "start": 3, "end": 9}``.
Entities = list[dict[str, Any]]
Tag = Callable[[str], Entities]
TagBatch = Callable[[list[str]], list[Entities]]

_WS = re.compile(r"\s+")
_BIO = re.compile(r"^[BIES]-")


def require_transformers() -> Any:
    """Import `transformers`, or raise a clear setup error.

    Returns:
        The `transformers` module.

    Raises:
        IntegrationError: The package is not installed. The message
            includes the `pip install "mimvo[hf]"` command.
    """
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise IntegrationError(
            "This detector needs the 'transformers' package: pip install 'mimvo[hf]'"
        ) from exc
    return transformers


def chunk_text(text: str, chunk_chars: int | None) -> list[str]:
    """Split `text` on whitespace into pieces a model can read.

    Classifiers see a fixed window, often 512 tokens. Scanning every chunk
    and keeping the worst score catches a payload buried at the end of a
    long document. Truncating to the first window would miss it.

    Args:
        text: The full memory content.
        chunk_chars: Maximum characters per chunk. `None`, or a limit that
            `text` already fits in, returns `[text]` unchanged.

    Returns:
        One or more chunks. Words are not split. The last chunk may be
        shorter. An empty input still returns one chunk, `[text]`.
    """
    if not chunk_chars or len(text) <= chunk_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0  # len(" ".join(current))
    for word in _WS.split(text.strip()):
        extra = len(word) + (1 if current else 0)
        if current and size + extra > chunk_chars:
            chunks.append(" ".join(current))
            current, size, extra = [], 0, len(word)
        current.append(word)
        size += extra
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


class HFTextClassifierDetector(BaseDetector):
    """A text classifier whose "bad" labels mean the security problem is present.

    Subclasses set `model_id` (the Hugging Face model name) and
    `positive_labels` (the label strings that count as a hit, compared in
    uppercase). The score is the highest positive-label probability across
    all chunks. A hit is emitted only when that score reaches `threshold`.

    Attributes:
        model_id: Hugging Face model id. Overridable in the constructor.
        positive_labels: Labels that count as a hit, stored uppercase.
        threshold: Minimum score, from 0 to 1. Default `0.5`.
        name: Detector name. Set `name=` to override the class default.
    """

    model_id: str = ""
    positive_labels: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        model_id: str | None = None,
        threshold: float = 0.5,
        positive_labels: Iterable[str] | None = None,
        classify: Classify | None = None,
        classify_batch: ClassifyBatch | None = None,
        device: int | str | None = None,
        max_length: int = 512,
        chunk_chars: int | None = 1500,
        batch_size: int = 32,
        name: str | None = None,
    ) -> None:
        """Load settings. The model weights are not downloaded yet.

        Args:
            model_id: Hugging Face model id. `None` uses the class `model_id`.
            threshold: Minimum positive-label score, from 0 to 1.
            positive_labels: Labels that count as a hit. `None` uses the
                class `positive_labels`. Compared after uppercasing.
            classify: Function `(text) -> [{"label": str, "score": float}, ...]`.
                Pass one in tests, or to call a remote model. `None` loads
                `transformers` on the first `detect` call.
            classify_batch: Function `(texts) -> [label scores per text]`.
                Used for batched scans. `None` uses the loaded pipeline in
                batches, or calls `classify` per text when you passed one.
            device: Device passed to the Hugging Face pipeline, such as `0`
                for the first GPU or `"cpu"`. `None` lets transformers choose.
            max_length: Token cap passed to the pipeline. Extra tokens in a
                chunk are truncated.
            chunk_chars: Split the memory into pieces of about this many
                characters before classifying. `None` sends the whole text.
            batch_size: Chunks per forward pass when the pipeline runs a
                batch of memories.
            name: Detector name written on each hit. `None` keeps the class name.

        Raises:
            ConfigurationError: `threshold` is outside 0 to 1, `model_id` is
                empty, `positive_labels` is empty, or `batch_size` is below 1.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError("threshold must be within [0, 1].")
        self.model_id = model_id or type(self).model_id
        if not self.model_id:
            raise ConfigurationError(f"{type(self).__name__} needs a model_id.")
        self.threshold = threshold
        source = positive_labels if positive_labels is not None else type(self).positive_labels
        self.positive_labels = frozenset(label.upper() for label in source)
        if not self.positive_labels:
            raise ConfigurationError(f"{type(self).__name__} needs positive_labels.")
        if batch_size < 1:
            raise ConfigurationError("batch_size must be >= 1.")
        self.device = device
        self.max_length = max_length
        self.chunk_chars = chunk_chars
        self.batch_size = batch_size
        if name:
            self.name = name
        self._classify: Classify | None = classify
        self._classify_batch: ClassifyBatch | None = classify_batch

    # -- inference -----------------------------------------------------------------

    def _load(self) -> tuple[Classify, ClassifyBatch]:
        transformers = require_transformers()
        kwargs: dict[str, Any] = {"model": self.model_id, "top_k": None}
        if self.device is not None:
            kwargs["device"] = self.device
        pipe = transformers.pipeline("text-classification", **kwargs)

        def classify(text: str) -> LabelScores:
            out = pipe(text, truncation=True, max_length=self.max_length)
            # ``top_k=None`` yields ``[[{label, score}, ...]]`` for a single input.
            if out and isinstance(out[0], list):
                return list(out[0])
            return list(out)

        def classify_batch(texts: list[str]) -> list[LabelScores]:
            out = pipe(
                texts, truncation=True, max_length=self.max_length, batch_size=self.batch_size
            )
            return [list(item) if isinstance(item, list) else [item] for item in out]

        return classify, classify_batch

    def classify(self, text: str) -> LabelScores:
        if self._classify is None:
            self._classify, self._classify_batch = self._load()
        return self._classify(text)

    def classify_many(self, texts: list[str]) -> list[LabelScores]:
        """Classify several texts, in one pipeline call when possible.

        Args:
            texts: The texts (chunks) to classify.

        Returns:
            Label scores per text, in order.
        """
        if self._classify_batch is None:
            if self._classify is not None:
                return [self._classify(text) for text in texts]
            self._classify, self._classify_batch = self._load()
        return self._classify_batch(texts)

    def _best_positive(self, outputs: Iterable[LabelScores]) -> tuple[float, str | None]:
        best, label = 0.0, None
        for scores in outputs:
            for item in scores:
                item_label = str(item.get("label", "")).upper()
                if item_label in self.positive_labels and float(item["score"]) > best:
                    best, label = float(item["score"]), item_label
        return best, label

    def _hits(self, score: float, label: str | None) -> list[Detection]:
        if label is None or score < self.threshold:
            return []
        return [self.hit(score=score, label=label.lower(), model=self.model_id)]

    def positive_score(self, text: str) -> tuple[float, str | None]:
        """Find the strongest "bad" label score in `text`.

        Args:
            text: Memory content. It is split with `chunk_chars` first.

        Returns:
            A pair `(score, label)`. `score` is the highest positive-label
            probability seen, from 0 to 1. `label` is that label in
            uppercase. When no positive label appears, the pair is `(0.0, None)`.
        """
        return self._best_positive(
            self.classify(chunk) for chunk in chunk_text(text, self.chunk_chars)
        )

    def detect_text(self, text: str) -> list[Detection]:
        return self._hits(*self.positive_score(text))

    def detect_batch(
        self, candidates: Sequence[MemoryCandidate], context: CheckContext
    ) -> list[list[Detection]]:
        """Classify every chunk of every memory in one batched pipeline call.

        Args:
            candidates: The memories, in scan order.
            context: Scan context. Unused.

        Returns:
            One list of detections per candidate, the same as `detect` gives.
        """
        chunks = [chunk_text(c.content, self.chunk_chars) for c in candidates]
        outputs = iter(self.classify_many([chunk for group in chunks for chunk in group]))
        return [
            self._hits(*self._best_positive([next(outputs) for _ in group])) for group in chunks
        ]


class HFTokenClassifierDetector(BaseDetector):
    """A token classifier that maps entity labels to finding codes.

    This is the NER style: the model marks spans such as a password or an
    email. `labels` maps an entity label (`"PASSWORD"`) to the finding code
    the parent check should emit (`"secret_detected"`). Any other label is
    ignored, so a personal-data model can be limited to credentials.

    Evidence lists the kinds of entity found. It never includes the matched
    text.

    Attributes:
        model_id: Hugging Face model id.
        default_labels: Class-level map of entity label to finding code.
            The constructor copies this when you do not pass `labels`.
        threshold: Minimum entity score, from 0 to 1.
    """

    model_id: str = ""
    default_labels: Mapping[str, str] = {}

    def __init__(
        self,
        *,
        model_id: str | None = None,
        labels: Mapping[str, str] | None = None,
        threshold: float = 0.5,
        tag: Tag | None = None,
        tag_batch: TagBatch | None = None,
        device: int | str | None = None,
        chunk_chars: int | None = 1000,
        batch_size: int = 32,
        name: str | None = None,
    ) -> None:
        """Load settings. The model weights are not downloaded yet.

        Args:
            model_id: Hugging Face model id. `None` uses the class `model_id`.
            labels: Map of entity label to finding code, for example
                `{"PASSWORD": "secret_detected"}`. `None` uses `default_labels`.
                Keys are compared in uppercase.
            threshold: Minimum entity score, from 0 to 1.
            tag: Function `(text) -> [{"entity_group": str, "score": float}, ...]`.
                Pass one in tests. `None` loads `transformers` on first use.
            tag_batch: Function `(texts) -> [entities per text]` for batched
                scans. `None` uses the loaded pipeline in batches, or calls
                `tag` per text when you passed one.
            device: Device passed to the pipeline, such as `"cpu"`. `None`
                lets transformers choose.
            chunk_chars: Split the memory into pieces of about this many
                characters. `None` sends the whole text.
            batch_size: Chunks per forward pass when the pipeline runs a
                batch of memories.
            name: Detector name written on each hit. `None` keeps the class name.

        Raises:
            ConfigurationError: `threshold` is outside 0 to 1, `model_id` is
                empty, `labels` is empty, or `batch_size` is below 1.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError("threshold must be within [0, 1].")
        self.model_id = model_id or type(self).model_id
        if not self.model_id:
            raise ConfigurationError(f"{type(self).__name__} needs a model_id.")
        source = labels if labels is not None else type(self).default_labels
        self.labels: dict[str, str] = {k.upper(): v for k, v in source.items()}
        if not self.labels:
            raise ConfigurationError(f"{type(self).__name__} needs a non-empty labels mapping.")
        if batch_size < 1:
            raise ConfigurationError("batch_size must be >= 1.")
        self.threshold = threshold
        self.device = device
        self.chunk_chars = chunk_chars
        self.batch_size = batch_size
        if name:
            self.name = name
        self._tag: Tag | None = tag
        self._tag_batch: TagBatch | None = tag_batch

    def _load(self) -> tuple[Tag, TagBatch]:
        transformers = require_transformers()
        kwargs: dict[str, Any] = {"model": self.model_id, "aggregation_strategy": "simple"}
        if self.device is not None:
            kwargs["device"] = self.device
        pipe = transformers.pipeline("token-classification", **kwargs)

        def tag_batch(texts: list[str]) -> list[Entities]:
            return [list(item) for item in pipe(texts, batch_size=self.batch_size)]

        return (lambda text: list(pipe(text))), tag_batch

    def tag(self, text: str) -> Entities:
        if self._tag is None:
            self._tag, self._tag_batch = self._load()
        return self._tag(text)

    def tag_many(self, texts: list[str]) -> list[Entities]:
        """Tag several texts, in one pipeline call when possible.

        Args:
            texts: The texts (chunks) to tag.

        Returns:
            Entities per text, in order.
        """
        if self._tag_batch is None:
            if self._tag is not None:
                return [self._tag(text) for text in texts]
            self._tag, self._tag_batch = self._load()
        return self._tag_batch(texts)

    def entity_kinds(self, text: str) -> dict[str, tuple[set[str], float]]:
        """Group recognized entities by the finding code they map to.

        Args:
            text: Memory content. It is split with `chunk_chars` first.

        Returns:
            A dict keyed by finding code. Each value is `(kinds, best_score)`:
            `kinds` is the set of entity labels seen (lowercased), and
            `best_score` is the highest score among them. Entities below
            `threshold`, or with a label that is not in `labels`, are left out.
        """
        return self._group(self.tag(chunk) for chunk in chunk_text(text, self.chunk_chars))

    def _group(self, tagged: Iterable[Entities]) -> dict[str, tuple[set[str], float]]:
        found: dict[str, tuple[set[str], float]] = {}
        for entities in tagged:
            for entity in entities:
                raw = str(entity.get("entity_group") or entity.get("entity") or "")
                label = _BIO.sub("", raw).upper()
                score = float(entity.get("score", 1.0))
                code = self.labels.get(label)
                if code is None or score < self.threshold:
                    continue
                kinds, best = found.get(code, (set(), 0.0))
                kinds.add(label.lower())
                found[code] = (kinds, max(best, score))
        return found

    def _hits(self, found: Mapping[str, tuple[set[str], float]]) -> list[Detection]:
        return [
            self.hit(code=code, score=best, kinds=sorted(kinds), model=self.model_id)
            for code, (kinds, best) in sorted(found.items())
        ]

    def detect_text(self, text: str) -> list[Detection]:
        return self._hits(self.entity_kinds(text))

    def detect_batch(
        self, candidates: Sequence[MemoryCandidate], context: CheckContext
    ) -> list[list[Detection]]:
        """Tag every chunk of every memory in one batched pipeline call.

        Args:
            candidates: The memories, in scan order.
            context: Scan context. Unused.

        Returns:
            One list of detections per candidate, the same as `detect` gives.
        """
        chunks = [chunk_text(c.content, self.chunk_chars) for c in candidates]
        outputs = iter(self.tag_many([chunk for group in chunks for chunk in group]))
        return [self._hits(self._group([next(outputs) for _ in group])) for group in chunks]


__all__ = [
    "Classify",
    "ClassifyBatch",
    "Entities",
    "HFTextClassifierDetector",
    "HFTokenClassifierDetector",
    "LabelScores",
    "Tag",
    "TagBatch",
    "chunk_text",
    "require_transformers",
]

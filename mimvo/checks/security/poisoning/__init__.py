"""Poisoning: false or attacker-controlled facts planted in memory or RAG.

`PoisoningCheck` runs one or more detectors:

* `HeuristicPoisoningDetector` — phrases that turn off a control, or that
  redirect payments and data. Default. Needs nothing extra.
* `TrustRAGDetector` — a tight cluster of near-copies among neighbouring
  records, using stored vectors and nearest-neighbour search. Default.
* `HubnessDetector` — records that appear in many other records' k-NN
  lists. Default. Needs stored embeddings.
* `RAGuardDetector` — chunk-wise perplexity plus a similarity filter.
  Needs `[hf]` or a `perplexity=` callback.
* `PerplexityDetector` — whole-text perplexity baseline. Needs `[hf]`.
* `EmbeddingConsistencyDetector` — re-embed the text and compare it with
  the stored vector. Needs an `embed` callback.
* `TemporalNLIDetector` — newer records that contradict older neighbours.
  Needs an `nli` callback.
* `ProbeQueryDetector` — a record that flips the answer to its own probe
  questions. Needs generate / retrieve / answer callbacks.
* `RevPRAGDetector` — activation probe for poisoned generations. Needs
  a `probe` callback and model weights.
"""

from __future__ import annotations

from .check import PoisoningCheck
from .consistency import EmbeddingConsistencyDetector
from .heuristic import HeuristicPoisoningDetector, poisoning_matches, redirect_matches
from .hubness import HubnessDetector
from .nli import TemporalNLIDetector
from .perplexity import PerplexityDetector
from .probe import ProbeQueryDetector
from .raguard import RAGuardDetector
from .revprag import RevPRAGDetector
from .trustrag import TrustRAGDetector

__all__ = [
    "EmbeddingConsistencyDetector",
    "HeuristicPoisoningDetector",
    "HubnessDetector",
    "PerplexityDetector",
    "PoisoningCheck",
    "ProbeQueryDetector",
    "RAGuardDetector",
    "RevPRAGDetector",
    "TemporalNLIDetector",
    "TrustRAGDetector",
    "poisoning_matches",
    "redirect_matches",
]

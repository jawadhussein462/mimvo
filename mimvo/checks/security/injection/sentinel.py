"""Qualifire Sentinel (`qualifire/prompt-injection-sentinel`).

A ModernBERT-large classifier with labels `benign` and `jailbreak`. It is
trained on instruction hijacking, role-play, and biased-content attacks.
The Sentinel paper (arXiv:2506.05446) reports 0.938 average F1 across four
public injection benchmarks, against 0.709 for the Protect AI DeBERTa
baseline. The 8k-token window means long documents need fewer chunks.

Install with `pip install "mimvo[hf]"`.
"""

from __future__ import annotations

from .._hf import HFTextClassifierDetector

SENTINEL = "qualifire/prompt-injection-sentinel"


class SentinelDetector(HFTextClassifierDetector):
    """Qualifire `prompt-injection-sentinel`. The label `jailbreak` is a hit.

    The constructor widens the window versus the base classifier:
    `max_length` defaults to 8192 tokens and `chunk_chars` to 20000.
    Pass either argument to override that. Other arguments come from
    `HFTextClassifierDetector`.
    """

    name = "sentinel"
    model_id = SENTINEL
    positive_labels = frozenset({"JAILBREAK", "INJECTION", "MALICIOUS", "LABEL_1"})

    def __init__(self, **kwargs: object) -> None:
        """Build a Sentinel detector with a long default context window.

        Args:
            **kwargs: Forwarded to `HFTextClassifierDetector`. `max_length`
                defaults to 8192 and `chunk_chars` to 20000 when you omit them.
        """
        kwargs.setdefault("max_length", 8192)
        kwargs.setdefault("chunk_chars", 20000)
        super().__init__(**kwargs)  # type: ignore[arg-type]


__all__ = ["SENTINEL", "SentinelDetector"]

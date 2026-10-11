"""Protect AI DeBERTa-v3 prompt-injection classifier.

`protectai/deberta-v3-base-prompt-injection-v2` (Apache-2.0) labels English
text `SAFE` or `INJECTION`. It is tuned for injected instructions, which
fits memory screening, and it does not cover jailbreak role-play or
non-English text. The same model powers LLM Guard's PromptInjection scanner.

Install with `pip install "mimvo[hf]"`.
"""

from __future__ import annotations

from .._hf import HFTextClassifierDetector

PROTECTAI_DEBERTA_V2 = "protectai/deberta-v3-base-prompt-injection-v2"


class ProtectAIDeBERTaDetector(HFTextClassifierDetector):
    """Protect AI `deberta-v3-base-prompt-injection-v2`.

    A hit is the label `INJECTION` (also accepted as `LABEL_1`) at or above
    `threshold`. Constructor arguments are the ones on
    `HFTextClassifierDetector`.
    """

    name = "protectai_deberta"
    model_id = PROTECTAI_DEBERTA_V2
    positive_labels = frozenset({"INJECTION", "LABEL_1"})


__all__ = ["PROTECTAI_DEBERTA_V2", "ProtectAIDeBERTaDetector"]

"""deepset DeBERTa-v3 injection classifier (`deepset/deberta-v3-base-injection`).

An earlier open classifier trained on `deepset/prompt-injections`. It labels
text `LEGIT` or `INJECTION`. It scores lower than Protect AI v2 on public
benchmarks, and it is cheap and permissively licensed. Use it as a second
voter with `InjectionCheck(min_detectors=2)`.

Install with `pip install "mimvo[hf]"`.
"""

from __future__ import annotations

from .._hf import HFTextClassifierDetector

DEEPSET_DEBERTA = "deepset/deberta-v3-base-injection"


class DeepsetDeBERTaDetector(HFTextClassifierDetector):
    """deepset `deberta-v3-base-injection`.

    A hit is the label `INJECTION` (also accepted as `LABEL_1`) at or above
    `threshold`. Constructor arguments are the ones on
    `HFTextClassifierDetector`.
    """

    name = "deepset_deberta"
    model_id = DEEPSET_DEBERTA
    positive_labels = frozenset({"INJECTION", "LABEL_1"})


__all__ = ["DEEPSET_DEBERTA", "DeepsetDeBERTaDetector"]

"""Meta Llama Prompt Guard 2 (`meta-llama/Llama-Prompt-Guard-2-86M`).

A classifier that labels text `BENIGN` or `MALICIOUS`. Malicious means an
explicit attempt to override earlier instructions, which is the pattern
Mimvo looks for in stored memory.

The 86M model handles non-English text. `Llama-Prompt-Guard-2-22M` is
faster and English-focused. Both are gated on the Hugging Face Hub and
see 512 tokens, so long memories are scanned in chunks.

Install with `pip install "mimvo[hf]"`, then:

    InjectionCheck(detectors=[HeuristicInjectionDetector(), PromptGuardDetector()])
"""

from __future__ import annotations

from .._hf import HFTextClassifierDetector

PROMPT_GUARD_2_86M = "meta-llama/Llama-Prompt-Guard-2-86M"
PROMPT_GUARD_2_22M = "meta-llama/Llama-Prompt-Guard-2-22M"


class PromptGuardDetector(HFTextClassifierDetector):
    """Llama Prompt Guard 2. The label `MALICIOUS` is a hit.

    `INJECTION`, `JAILBREAK`, and `LABEL_1` are also accepted. Pass
    `model_id=PROMPT_GUARD_2_22M` to use the smaller English model.
    Other constructor arguments come from `HFTextClassifierDetector`.
    """

    name = "prompt_guard"
    model_id = PROMPT_GUARD_2_86M
    positive_labels = frozenset({"MALICIOUS", "INJECTION", "JAILBREAK", "LABEL_1"})


__all__ = ["PROMPT_GUARD_2_22M", "PROMPT_GUARD_2_86M", "PromptGuardDetector"]

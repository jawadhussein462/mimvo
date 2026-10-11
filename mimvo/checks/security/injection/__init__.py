"""Injection: instructions aimed at the agent that were saved as memory.

`InjectionCheck` runs one or more detectors:

* `HeuristicInjectionDetector` — phrase patterns after deobfuscation. Default.
  Needs nothing extra.
* `KnownAnswerDetector` — canary instruction; needs a `complete` callback.
* `DataSentinelDetector` — two-step known-answer game; needs `complete`.
* `EmbeddingInjectionDetector` — classifier on the stored vector.
* `AttentionTrackerDetector`, `TaskTrackerDetector`, `PIShieldDetector` —
  injectable LLM-internal probes.
* Chat-prompt baselines (optional): Prompt Guard 2, ProtectAI/deepset
  DeBERTa, Sentinel, Lakera Guard, Azure Prompt Shields. Trained on
  user-to-chat jailbreaks, not stored memory, so expect distribution shift.
"""

from __future__ import annotations

from .check import InjectionCheck
from .datasentinel import DataSentinelDetector
from .deepset_deberta import DeepsetDeBERTaDetector
from .embedding_clf import EmbeddingInjectionDetector
from .heuristic import HeuristicInjectionDetector, injection_matches
from .known_answer import KnownAnswerDetector
from .lakera_guard import LakeraGuardDetector
from .probes import AttentionTrackerDetector, PIShieldDetector, TaskTrackerDetector
from .prompt_guard import PromptGuardDetector
from .prompt_shield import PromptShieldDetector
from .protectai_deberta import ProtectAIDeBERTaDetector
from .sentinel import SentinelDetector

__all__ = [
    "AttentionTrackerDetector",
    "DataSentinelDetector",
    "DeepsetDeBERTaDetector",
    "EmbeddingInjectionDetector",
    "HeuristicInjectionDetector",
    "InjectionCheck",
    "KnownAnswerDetector",
    "LakeraGuardDetector",
    "PIShieldDetector",
    "PromptGuardDetector",
    "PromptShieldDetector",
    "ProtectAIDeBERTaDetector",
    "SentinelDetector",
    "TaskTrackerDetector",
    "injection_matches",
]

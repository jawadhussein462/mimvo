"""Tests for secrets detectors.

Model and hosted detectors are given fake callables, so these tests
download nothing and open no network connection.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mimvo import Mimvo
from mimvo.checks.security import SecretsCheck
from mimvo.checks.security.secrets import (
    GLINER2_ALL_LABELS,
    PIIRANHA_ALL_LABELS,
    PRESIDIO_ALL_LABELS,
    STARPII_ALL_LABELS,
    DetectSecretsDetector,
    EntropyDetector,
    GLiNER2PIIDetector,
    GLiNERPIIDetector,
    HeuristicSecretsDetector,
    PiiranhaDetector,
    PresidioDetector,
    StarPIIDetector,
    secret_kinds,
    shannon_entropy,
)
from mimvo.exceptions import ConfigurationError
from mimvo.models.enums import Action, Severity
from tests.factories import make_candidate, make_context

KEY = "sk-abcdefghijklmnop1234567890"
CLEAN = "Alice prefers annual billing."


# -- heuristic ------------------------------------------------------------------------------


def test_heuristic_detector_reports_kinds_not_values():
    (d,) = HeuristicSecretsDetector().detect_text(f"my key is {KEY}")
    assert d.code == "secret_detected" and d.evidence["kinds"] == secret_kinds(f"my key is {KEY}")
    assert KEY not in str(d)
    assert HeuristicSecretsDetector().detect_text(CLEAN) == []


# -- entropy --------------------------------------------------------------------------------


def test_shannon_entropy():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == pytest.approx(2.0)


def test_entropy_detector_flags_random_tokens_not_words():
    det = EntropyDetector()
    (d,) = det.detect_text("session token: 8fK2mQ9xLp4vRn7tWc1yZb6hJd3sGa0eUi5oPk")
    assert d.evidence["kinds"] == ["high_entropy_base64"] and d.evidence["entropy"] > 4.5
    (h,) = det.detect_text("commit 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b")
    assert h.evidence["kinds"] == ["high_entropy_hex"]
    assert det.detect_text("internationalization and localization are long words") == []
    assert det.detect_text(CLEAN) == []


def test_entropy_detector_never_leaks_the_token():
    token = "8fK2mQ9xLp4vRn7tWc1yZb6hJd3sGa0eUi5oPk"
    (d,) = EntropyDetector().detect_text(f"token {token}")
    assert token not in str(d.evidence)


@pytest.mark.parametrize("kwargs", [{"base64_limit": 9}, {"hex_limit": -1}, {"min_length": 4}])
def test_entropy_validation(kwargs):
    with pytest.raises(ConfigurationError):
        EntropyDetector(**kwargs)


# -- Hugging Face token classifiers ---------------------------------------------------------


def _entities(*items):
    return [{"entity_group": label, "score": score, "word": "REDACTED"} for label, score in items]


def test_piiranha_default_labels_block_credentials_only():
    det = PiiranhaDetector(tag=lambda t: _entities(("I-PASSWORD", 0.99), ("I-GIVENNAME", 0.95)))
    (d,) = det.detect_text("My name is Alice and my password is hunter2")
    assert d.code == "secret_detected" and d.evidence["kinds"] == ["password"]
    assert "REDACTED" not in str(d.evidence)


def test_piiranha_all_labels_add_pii_review():
    det = PiiranhaDetector(
        tag=lambda t: _entities(("PASSWORD", 0.99), ("GIVENNAME", 0.95), ("EMAIL", 0.9)),
        labels=PIIRANHA_ALL_LABELS,
    )
    codes = {d.code: d for d in det.detect_text("x")}
    assert codes["secret_detected"].evidence["kinds"] == ["password"]
    assert codes["pii_detected"].evidence["kinds"] == ["email", "givenname"]
    chk = SecretsCheck(detectors=[det])
    findings = {f.code: f for f in chk.check(make_candidate("x"), make_context())}
    assert findings["secret_detected"].recommended_action == Action.DELETE
    assert findings["pii_detected"].recommended_action == Action.REVIEW
    assert findings["pii_detected"].severity == Severity.MEDIUM


def test_token_classifier_threshold_and_unknown_labels():
    det = PiiranhaDetector(tag=lambda t: _entities(("PASSWORD", 0.3), ("FOO", 0.99)))
    assert det.detect_text("x") == []


def test_starpii_defaults_to_keys_and_passwords():
    det = StarPIIDetector(tag=lambda t: _entities(("KEY", 0.9), ("USERNAME", 0.9)))
    (d,) = det.detect_text("x")
    assert d.evidence["kinds"] == ["key"]
    assert set(STARPII_ALL_LABELS) == {"KEY", "PASSWORD", "EMAIL", "NAME", "IP_ADDRESS", "USERNAME"}


# -- GLiNER2 / GLiNER -----------------------------------------------------------------------


def test_gliner2_requests_only_mapped_labels_and_maps_codes():
    seen = {}

    def extract(text, labels):
        seen["labels"] = labels
        return {
            "entities": {
                "api_key": [{"text": "REDACTED", "confidence": 0.93}],
                "email": [{"text": "REDACTED", "confidence": 0.99}],
                "person": [],
            }
        }

    det = GLiNER2PIIDetector(extract=extract, labels=GLINER2_ALL_LABELS)
    codes = {d.code: d for d in det.detect_text("x")}
    assert seen["labels"] == sorted(GLINER2_ALL_LABELS)
    assert codes["secret_detected"].evidence["kinds"] == ["api_key"]
    assert codes["pii_detected"].evidence["kinds"] == ["email"]
    assert "REDACTED" not in str(codes)


def test_gliner2_default_scope_is_credentials():
    det = GLiNER2PIIDetector(
        extract=lambda t, labels: {"entities": {"email": [{"text": "x", "confidence": 0.9}]}}
    )
    assert "email" not in det.labels
    assert det.detect_text("x") == []


def test_gliner_pii_detector():
    det = GLiNERPIIDetector(
        predict=lambda t, labels: [
            {"label": "credit card number", "score": 0.97, "text": "REDACTED"},
            {"label": "person", "score": 0.9, "text": "REDACTED"},
            {"label": "iban", "score": 0.2, "text": "REDACTED"},
        ]
    )
    (d,) = det.detect_text("x")
    assert d.code == "secret_detected" and d.evidence["kinds"] == ["credit card number"]
    assert d.score == pytest.approx(0.97)


@pytest.mark.parametrize("cls", [GLiNER2PIIDetector, GLiNERPIIDetector, PresidioDetector])
def test_label_map_detectors_validate(cls):
    with pytest.raises(ConfigurationError):
        cls(labels={})
    with pytest.raises(ConfigurationError):
        cls(threshold=2)


# -- Presidio -------------------------------------------------------------------------------


def test_presidio_detector_maps_entity_types():
    def analyze(text, entities):
        assert entities == sorted(PRESIDIO_ALL_LABELS)
        return [
            SimpleNamespace(entity_type="CREDIT_CARD", score=0.95),
            SimpleNamespace(entity_type="PERSON", score=0.85),
            SimpleNamespace(entity_type="PHONE_NUMBER", score=0.4),
        ]

    det = PresidioDetector(analyze=analyze, labels=PRESIDIO_ALL_LABELS)
    codes = {d.code: d for d in det.detect_text("x")}
    assert codes["secret_detected"].evidence["kinds"] == ["credit_card"]
    assert codes["pii_detected"].evidence["kinds"] == ["person"]


def test_presidio_accepts_a_configured_analyzer():
    class FakeAnalyzer:
        def analyze(self, *, text, language, entities, score_threshold):
            assert language == "de" and score_threshold == 0.5
            return [SimpleNamespace(entity_type="IBAN_CODE", score=0.99)]

    det = PresidioDetector(analyzer=FakeAnalyzer(), language="de")
    (d,) = det.detect_text("x")
    assert d.evidence["kinds"] == ["iban_code"]


# -- detect-secrets -------------------------------------------------------------------------


def test_detect_secrets_reports_plugin_types_as_kinds():
    det = DetectSecretsDetector(scan=lambda t: {"AWS Access Key", "Secret Keyword"})
    (d,) = det.detect_text("x")
    assert d.evidence["kinds"] == ["aws_access_key", "secret_keyword"]
    excluded = DetectSecretsDetector(
        scan=lambda t: {"Secret Keyword"}, exclude_types=["Secret Keyword"]
    )
    assert excluded.detect_text("x") == []


# -- stacking through the facade ------------------------------------------------------------


def test_stacked_secrets_check_blocks_and_never_leaks():
    token = "8fK2mQ9xLp4vRn7tWc1yZb6hJd3sGa0eUi5oPk"
    guard = Mimvo(checks=[SecretsCheck(detectors=[HeuristicSecretsDetector(), EntropyDetector()])])
    report = guard.scan([{"id": "s", "content": f"Deploy with {KEY} and session {token}"}])
    secret = next(f for f in report.findings if f.code == "secret_detected")
    assert secret.recommended_action == Action.DELETE
    assert secret.detectors == ["entropy", "heuristic"]
    blob = report.model_dump_json()
    assert KEY not in blob and token not in blob
    assert guard.scan([{"id": "c", "content": CLEAN}]).clean

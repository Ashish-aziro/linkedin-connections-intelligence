from __future__ import annotations

from app.services import company_intel


def test_company_key_prefers_id_over_name():
    assert company_intel.company_key("1441", "Google") == "id:1441"
    assert company_intel.company_key(None, "Google LLC") == "name:google"


def test_unclassified_company_is_unknown_not_false(db):
    out = company_intel.get_or_classify(db, [(None, "Some Totally Unknown Startup Name Xyz123", None)])
    key = company_intel.company_key(None, "Some Totally Unknown Startup Name Xyz123")
    assert out[key]["is_startup"] is None  # UNKNOWN, never coerced to False
    assert out[key]["is_big_tech"] is None
    assert out[key]["provenance"] in ("unknown", "ai_company_inference")


def test_classification_is_cached_not_repeated(db, monkeypatch):
    calls = {"n": 0}

    from app.schemas import CompanyClassificationBatch, CompanyClassificationItem

    def fake_generate(system, user, schema, **kw):  # noqa: ARG001
        calls["n"] += 1
        return (
            CompanyClassificationBatch(
                companies=[
                    CompanyClassificationItem(
                        key="id:1441",
                        industries=["technology", "internet"],
                        categories=["big_tech"],
                        is_technology_company=True,
                        is_big_tech=True,
                        is_startup=False,
                        confidence=0.97,
                        reason="Well-known large technology company.",
                    )
                ]
            ),
            "anthropic:paid",
            "test-model",
        )

    monkeypatch.setattr("app.services.company_intel.generate_structured", fake_generate)

    r1 = company_intel.get_or_classify(db, [("1441", "Google", "https://linkedin.com/company/google")])
    assert calls["n"] == 1
    assert r1["id:1441"]["is_big_tech"] is True
    assert r1["id:1441"]["is_startup"] is False
    assert r1["id:1441"]["provenance"] == "ai_company_inference"

    # second call for the SAME company must NOT re-invoke the LLM
    r2 = company_intel.get_or_classify(db, [("1441", "Google", "https://linkedin.com/company/google")])
    assert calls["n"] == 1
    assert r2["id:1441"]["is_big_tech"] is True


def test_disabled_classification_returns_unknown_without_calling_llm(db, monkeypatch):
    monkeypatch.setattr("app.config.settings.company_classification_enabled", False)
    called = {"n": 0}
    monkeypatch.setattr(
        "app.services.company_intel.generate_structured",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call LLM")),
    )
    out = company_intel.get_or_classify(db, [(None, "Some Company", None)])
    key = company_intel.company_key(None, "Some Company")
    assert out[key]["is_startup"] is None
    assert called["n"] == 0

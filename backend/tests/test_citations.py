from app.services.citations import validate_citations

ALLOWED = {"S1", "S2", "S3"}


def test_valid_citations_pass_unchanged():
    check = validate_citations("Sign out and back in [S1]. Wait 30 minutes [S2].", ["S1", "S2"], ALLOWED)
    assert check.valid_ids == ["S1", "S2"]
    assert check.invalid_ids == []
    assert check.cleaned_answer == "Sign out and back in [S1]. Wait 30 minutes [S2]."
    assert check.warnings == []


def test_invented_citation_is_removed_and_reported():
    check = validate_citations("Refunds take 3 days [S9]. Contact us [S1].", ["S9", "S1"], ALLOWED)
    assert check.valid_ids == ["S1"]
    assert check.invalid_ids == ["S9"]
    assert "[S9]" not in check.cleaned_answer
    assert check.cleaned_answer == "Refunds take 3 days. Contact us [S1]."
    assert any("S9" in w for w in check.warnings)


def test_inline_ids_count_even_if_not_declared():
    check = validate_citations("Use the reset link [S3].", [], ALLOWED)
    assert check.valid_ids == ["S3"]


def test_answer_without_any_citation_is_flagged():
    check = validate_citations("Just restart your router.", [], ALLOWED)
    assert check.valid_ids == []
    assert any("cites no retrieved source" in w for w in check.warnings)


def test_declared_but_never_marked_inline_is_flagged():
    check = validate_citations("Restart the app.", ["S1"], ALLOWED)
    assert check.valid_ids == ["S1"]
    assert any("no sentence is marked" in w for w in check.warnings)

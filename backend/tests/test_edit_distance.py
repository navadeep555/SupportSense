from app.services.edit_distance import edit_ratio, strip_citation_markers, word_edit_distance


def test_levenshtein_on_words():
    assert word_edit_distance("a b c".split(), "a b c".split()) == 0
    assert word_edit_distance("a b c".split(), "a x c".split()) == 1      # replace
    assert word_edit_distance("a b c".split(), "a c".split()) == 1        # delete
    assert word_edit_distance([], "a b".split()) == 2                     # insert all


def test_citation_markers_are_stripped_for_the_customer():
    assert strip_citation_markers("Sign out [S1]. Wait 30 minutes [S2].") == "Sign out. Wait 30 minutes."


def test_unchanged_except_markers_and_case_is_zero():
    assert edit_ratio("Sign out and back in [S1].", "sign out and back in.") == 0.0


def test_ratio_is_normalised_by_longer_text():
    # 7 words vs 5 words: 2 inserted ("please", "sign") -> 2/7
    assert edit_ratio("Sign out and back in [S1].", "Please sign out and sign back in.") == round(2 / 7, 4)


def test_complete_rewrite_is_one():
    assert edit_ratio("alpha beta", "gamma delta") == 1.0

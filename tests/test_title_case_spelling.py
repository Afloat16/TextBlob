import pytest

from textblob import TextBlob, Word
from textblob._text import Spelling


@pytest.fixture
def spelling():
    model = Spelling()
    model.update({"good": 40, "wood": 20, "food": 10, "spelling": 100, "banana": 30})
    return model


@pytest.mark.parametrize("word", ["Good", "Wood", "Spelling", "Banana"])
def test_known_title_case_word_has_full_confidence(spelling, word):
    assert spelling.suggest(word) == [(word, 1.0)]


@pytest.mark.parametrize("word", ["Godo", "Speling", "Spelingg", "Bannanna"])
def test_title_case_uses_the_same_edit_budget_and_probabilities(spelling, word):
    expected = [
        (candidate.title(), probability)
        for candidate, probability in spelling.suggest(word.lower())
    ]
    assert spelling.suggest(word) == expected
    assert all(probability > 0 for _, probability in expected)


def test_explicitly_cased_lexicon_entry_remains_preferred(spelling):
    spelling["Good"] = 3
    assert spelling.suggest("Good") == [("Good", 1.0)]


@pytest.mark.parametrize("word", ["Lnodon", "Lnodonn"])
def test_custom_proper_name_corrections_are_retained(word):
    spelling = Spelling()
    spelling.update({"London": 10})
    assert spelling.suggest(word) == [("London", 1.0)]


@pytest.mark.parametrize("word", ["Qqvzzx", "İqvzzx", "SPELING", "sPeLiNg"])
def test_unknown_and_other_case_styles_keep_the_original_token(spelling, word):
    assert spelling.suggest(word) == [(word, 0.0)]


@pytest.mark.parametrize("word", ["I", "A", "!", "\n", "12.34"])
def test_special_tokens_are_unchanged(spelling, word):
    assert spelling.suggest(word) == [(word, 1.0)]


@pytest.mark.parametrize("word", ["Good", "Word", "Bannanna", "Spelingg"])
def test_bundled_english_model_matches_lowercase_suggestions(word):
    expected = [
        (candidate.title(), probability)
        for candidate, probability in Word(word.lower()).spellcheck()
    ]
    assert Word(word).spellcheck() == expected


def test_blob_correction_handles_title_case_two_edit_words():
    assert TextBlob("Bannanna and Spelingg.").correct() == "Banana and Spelling."

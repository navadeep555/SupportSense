import pytest

from app.services.pii import luhn_valid, mask_pii


def test_masks_email_and_phone():
    result = mask_pii("Mail ravi.k@gmail.com or call +91 98765 43210.")
    assert result.text == "Mail [EMAIL] or call [PHONE]."
    assert result.found == {"email": 1, "phone": 1}


@pytest.mark.parametrize("phone", ["(555) 123-4567", "555-123-4567", "+1 (555) 123-4567", "9876543210"])
def test_phone_formats(phone):
    assert mask_pii(f"call {phone} now").text == "call [PHONE] now"


def test_card_needs_valid_luhn_checksum():
    assert mask_pii("card 4111 1111 1111 1111 please").text == "card [CARD] please"
    # 16 digits that fail Luhn: probably not a card number, left alone (documented trade-off)
    assert mask_pii("ref 1234 5678 9012 3456").found == {}


def test_password_after_keyword_is_masked():
    result = mask_pii("My password is Hunter2! and PASSWORD: abc123")
    assert result.text == "My password is [PASSWORD] and PASSWORD: [PASSWORD]"
    assert result.found == {"password": 2}


@pytest.mark.parametrize("text", [
    "Error SN-2003 on my TV",          # error codes are needed for retrieval
    "charged on 2026-03-14",           # dates
    "refund within 14 days, order 12345, amount 499.00",
])
def test_support_details_are_not_masked(text):
    result = mask_pii(text)
    assert result.text == text and result.found == {}


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")

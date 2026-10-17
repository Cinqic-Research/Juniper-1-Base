"""Planted-credential and PII tests.

Fake credentials are assembled at run time from fragments so that no
credential-shaped literal exists in the repository for scanners to flag.
"""

import pytest

from j1data import privacy


def fake(prefix: str, alphabet: str, length: int) -> str:
    body = (alphabet * (length // len(alphabet) + 1))[:length]
    return prefix + body


ALNUM = "Q7wErT9yUiOp2AsDfGhJ4kLzXcVbNm3"

PLANTED = {
    "aws_access_key": "key id " + fake("AK" + "IA", "QZXWVUTSRPONMLKJ", 16),
    "github_token": "token: " + fake("gh" + "p_", ALNUM, 36),
    "slack_token": fake("xo" + "xb-", "1234567890-abcdefABCDEF", 30),
    "google_api_key": fake("AI" + "za", ALNUM + "_-", 35),
    "openai_key": "OPENAI_API_KEY=" + fake("s" + "k-proj-", ALNUM, 40),
    "private_key": "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIEow...\n-----END RSA PRIVATE KEY-----",
    "credential_url": "DATABASE_URL=postgres://app_user:" + "Zr8qL2mN5vX1@db.internal.corp:5432/prod",
    "password_assignment": 'password = "' + "Tq9zK2mP7vXr4L" + '"',
    "auth_header": "Authorization: Bearer " + fake("", ALNUM, 40) + "9",
    "jwt": "eyJhbGciOiJIUzI1NiJ9." + "eyJ1c2VyIjoiamRvZTEyMyJ9" + "." + fake("", ALNUM, 30),
}


@pytest.mark.parametrize("kind", sorted(PLANTED))
def test_planted_credentials_reject_the_document(kind: str) -> None:
    text = f"Some surrounding text.\n{PLANTED[kind]}\nMore text after."
    result = privacy.scrub(text)
    assert f"privacy.secret.{kind}" in result.reject_reasons


@pytest.mark.parametrize(
    "text",
    [
        "AWS documentation uses AKIAIOSFODNN7EXAMPLE as the sample key id.",
        'password = "your_password_here"',
        "postgres://user:password@localhost/db",
        "postgres://${DB_USER}:${DB_PASS}@host/db",
        "api_key = '<YOUR-API-KEY>'",
        "token: xxxxxxxxxxxxxxxxxxxx",
        "The jwt.io example eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ.abcdefghijklmn",
    ],
)
def test_documented_placeholders_are_not_secrets(text: str) -> None:
    assert privacy.find_secrets(text) == []


def test_reasons_and_results_never_contain_the_secret_value() -> None:
    secret = PLANTED["github_token"]
    result = privacy.scrub(secret)
    assert all(secret not in reason for reason in result.reject_reasons)


def test_email_phone_ip_and_home_path_are_redacted() -> None:
    text = (
        "Contact jane.doe@university.edu or call (415) 867-5309. Server 203.0.113.7 is a doc address; "
        "the box at 54.23.198.77 logged in. Files live in /home/jdoe/projects and "
        "C:\\Users\\JaneDoe\\Desktop."
    )
    result = privacy.scrub(text)
    assert not result.reject_reasons
    assert "jane.doe@university.edu" not in result.text and privacy.EMAIL_PLACEHOLDER in result.text
    assert "867-5309" not in result.text and privacy.PHONE_PLACEHOLDER in result.text
    assert "54.23.198.77" not in result.text and privacy.IPV4_PLACEHOLDER in result.text
    assert "/home/jdoe" not in result.text and "/home/user/projects" in result.text
    assert "JaneDoe" not in result.text
    assert result.redactions["email"] == 1 and result.redactions["phone"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "Use 8.8.8.8 or 1.1.1.1 for DNS; loopback is 127.0.0.1 and LAN is 192.168.1.20.",
        "Upgrade to version 4.12.10.3 or v 10.0.0.120 of the driver.",
        "Write to user@example.com or noreply@example.org.",
        "The numbers 415 867 5309 appear in this table of exercise data.",
        "Paths like /home/user/.bashrc and /Users/Shared are generic.",
    ],
)
def test_non_personal_values_are_left_alone(text: str) -> None:
    result = privacy.scrub(text)
    assert result.text == text and not result.reject_reasons


def test_payment_cards_and_ssn() -> None:
    assert privacy.find_high_risk_pii("Card: 4539 1488 0343 6467 exp 12/27") == ["payment_card"]
    assert privacy.find_high_risk_pii("Use the test card 4242 4242 4242 4242 in sandbox mode.") == []
    assert privacy.find_high_risk_pii("Card: 4539 1488 0343 6468 is not Luhn-valid") == []
    assert privacy.find_high_risk_pii("Sequence 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16") == []
    assert privacy.find_high_risk_pii("SSN: 123-45-6789") == ["us_ssn"]
    assert privacy.find_high_risk_pii("Call 123-45-6789 later") == []


def test_phone_redaction_disabled_for_code() -> None:
    code = "TABLE = ['212-555-0187', '(212) 867-5309']"
    assert privacy.scrub(code, redact_phones=False).text == code


def test_numeric_tables_without_card_context_are_not_cards() -> None:
    table = "Results: 4539 1488 0343 6467 and 1000 2000 3000 4000 for the four runs."
    assert privacy.find_high_risk_pii(table) == []

"""Secret and personal-information handling.

Policy (j1.privacy.v1):

* Credentials (provider keys, private keys, tokens, credentialed connection
  strings, authorization headers, high-entropy password assignments) and
  government or payment identifiers reject the whole document. Partial
  redaction of a leaked credential still teaches its shape and context.
* E-mail addresses, phone numbers, public IPv4 addresses and user names inside
  home-directory paths are replaced with fixed, obviously synthetic values
  (the FineWeb convention for e-mail and IP), because they are usually
  incidental to otherwise useful text.

Matched values are never returned, logged or stored; callers receive counts
and reason codes only.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass, field

VERSION = "j1.privacy.v1"

EMAIL_PLACEHOLDER = "email@example.com"
PHONE_PLACEHOLDER = "555-555-0100"
IPV4_PLACEHOLDER = "192.0.2.1"
PATH_USER_PLACEHOLDER = "user"

_SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----"
    ),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "aws_secret_key": re.compile(
        r"aws_?secret_?(?:access_?)?key[\"']?\s*[:=]\s*[\"']?(?P<value>[A-Za-z0-9/+=]{40})\b", re.IGNORECASE
    ),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b"),
    "gitlab_token": re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
    "slack_token": re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"),
    "slack_webhook": re.compile(r"hooks\.slack\.com/services/T[A-Z0-9]{6,}/B[A-Z0-9]{6,}/[A-Za-z0-9]{20,}"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "stripe_key": re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{20,}\b"),
    "openai_key": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{32,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z]{34}\b"),
    "sendgrid_key": re.compile(r"\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b"),
    "twilio_key": re.compile(r"\bSK[0-9a-f]{32}\b"),
    "npm_token": re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "auth_header": re.compile(
        r"Authorization\s*:\s*(?:Bearer|Basic|Token)\s+(?P<value>[A-Za-z0-9._~+/=\-]{24,})", re.IGNORECASE
    ),
    "credential_url": re.compile(
        r"\b[a-z][a-z0-9+.\-]{1,20}://(?P<user>[^\s:/@'\"<>]{1,64}):(?P<value>[^\s@/'\"<>]{6,128})@[A-Za-z0-9.\-]+",
        re.IGNORECASE,
    ),
    # Anchored on the quoted value; the key name is checked on the preceding text
    # (see find_secrets), which avoids a case-insensitive alternation at every position.
    "password_assignment": re.compile(r"[:=][ \t]*[\"'](?P<value>[^\"'\s]{12,})[\"']"),
}

# Literal substrings at least one of which must occur (case-sensitively, or in the
# lower-cased text for entries in _LOWER_TRIGGERS) before a pattern is run; this
# keeps the per-document cost to a few substring searches for clean text.
_TRIGGERS: dict[str, tuple[str, ...]] = {
    "private_key": ("PRIVATE KEY",),
    "aws_access_key": ("AKIA", "ASIA"),
    "github_token": ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"),
    "gitlab_token": ("glpat-",),
    "slack_token": ("xox",),
    "slack_webhook": ("hooks.slack.com",),
    "google_api_key": ("AIza",),
    "stripe_key": ("_live_",),
    "openai_key": ("sk-",),
    "anthropic_key": ("sk-ant-",),
    "huggingface_token": ("hf_",),
    "sendgrid_key": ("SG.",),
    "twilio_key": ("SK",),
    "npm_token": ("npm_",),
    "jwt": ("eyJ",),
    "credential_url": ("://",),
}
_LOWER_TRIGGERS: dict[str, tuple[str, ...]] = {
    "aws_secret_key": ("aws",),
    "auth_header": ("authorization",),
    "password_assignment": ("pass", "pwd", "secret", "key", "token"),
}

_PASSWORD_KEY_RE = re.compile(
    r"(?:password|passwd|pwd|secret|api_?key|access_?token|auth_?token|secret_?key|client_?secret)"
    r"[\"']?[ \t]*$"
)

_HIGH_RISK_PII: dict[str, re.Pattern[str]] = {
    "us_ssn": re.compile(
        r"(?:ssn|social security(?: number| no\.?)?)\D{0,20}"
        r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
        re.IGNORECASE,
    ),
    # Grouped card numbers (4-4-4-4[-3], Amex 4-6-5) with one consistent separator.
    "payment_card": re.compile(r"\b\d{4}([ -])\d{4}\1\d{4}\1\d{4}(?:\1\d{3})?\b|\b\d{4}([ -])\d{6}\2\d{5}\b"),
    # An unseparated 13-19 digit run shortly after a card keyword (run only when a keyword occurs).
    "payment_card_keyword": re.compile(r"(?:card|visa|mastercard|amex|credit|debit)\D{0,30}\b(\d{13,19})\b"),
}

_CARD_KEYWORDS = ("card", "visa", "mastercard", "amex", "credit", "debit", "cvv", "expir")
_CARD_KEYWORD_RE = _HIGH_RISK_PII["payment_card_keyword"]

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]{1,64}@(?P<domain>[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,24})\b"
)
# North American numbers written with phone punctuation; space-only digit groups
# are excluded because they are usually table or exercise data.
_PHONE_RE = re.compile(
    r"(?<![\w.+\-])(?:\+?1[ .\-])?(?:\([2-9]\d{2}\) ?|[2-9]\d{2}[.\-])\d{3}[.\-]\d{4}(?![\w\-])"
)
# Octet ranges are validated by ipaddress in the callback; a simple pattern scans math-heavy text fast.
_IPV4_RE = re.compile(r"(?<![\d.])\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?![\d.])")
_HOME_PATH_RE = re.compile(
    r"(?P<prefix>/home/|/Users/|[A-Za-z]:\\Users\\|[A-Za-z]:/Users/)(?P<user>[A-Za-z0-9._\-]{2,32})"
)

_PLACEHOLDER_VALUE_RE = re.compile(
    r"^(?:x+|\*+|\.+|<.*>|\$\{.*\}|\{\{.*\}\}|%\(.*\)s|your[_\-]?.*|.*example.*|.*placeholder.*|"
    r"changeme|password|passw0rd|secret|test.*|dummy.*|sample.*|fake.*|redacted|none|null|undefined)$",
    re.IGNORECASE,
)
_SAFE_EMAIL_DOMAINS = re.compile(
    r"(?:^|\.)(?:example\.(?:com|org|net)|test|invalid|localhost|example)$", re.IGNORECASE
)
_GENERIC_USERS = frozenset(
    {
        "user",
        "username",
        "you",
        "yourname",
        "your_name",
        "me",
        "name",
        "example",
        "admin",
        "ubuntu",
        "pi",
        "root",
        "runner",
        "vagrant",
        "ec2-user",
        "jovyan",
        "shared",
        "public",
        "guest",
        "default",
        "docker",
        "foo",
        "bar",
        "john",
        "jane",
        "alice",
        "bob",
        "someone",
        "myuser",
        "my_user",
        "your-username",
        "your_username",
        "USER",
        "USERNAME",
        "Shared",
        "Public",
        "Default",
        "All",
    }
)
# Payload of the jwt.io sample token ({"sub": "1234567890", "name": "John Doe", ...}).
_DOCUMENTATION_JWT_PAYLOAD = "eyJzdWIiOiIxMjM0NTY3ODkw"
_VERSION_CONTEXT_RE = re.compile(r"(?:\bv|version|ver\.?|release|build|rev\.?)\s*$", re.IGNORECASE)
_WELL_KNOWN_IPS = frozenset(
    {"8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1", "9.9.9.9", "208.67.222.222", "208.67.220.220"}
)
# Published processor test cards appear in countless payment tutorials.
_TEST_CARDS = frozenset(
    {
        "4111111111111111",
        "4242424242424242",
        "5555555555554444",
        "378282246310005",
        "6011111111111117",
        "4012888888881881",
        "5105105105105100",
        "4000056655665556",
        "371449635398431",
        "30569309025904",
    }
)


@dataclass
class PrivacyResult:
    text: str
    reject_reasons: list[str] = field(default_factory=list)
    redactions: Counter[str] = field(default_factory=Counter)


def _shannon_entropy(value: str) -> float:
    counts = Counter(value)
    total = len(value)
    return -sum(c / total * math.log2(c / total) for c in counts.values())


def _luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _looks_like_real_secret(kind: str, match: re.Match[str]) -> bool:
    text = match.group(0)
    if "EXAMPLE" in text.upper() and kind != "private_key":
        return False
    if kind == "jwt" and _DOCUMENTATION_JWT_PAYLOAD in text:
        return False
    value = match.groupdict().get("value")
    if value is None:
        return True
    if _PLACEHOLDER_VALUE_RE.match(value):
        return False
    if kind == "credential_url":
        user = match.group("user")
        if _PLACEHOLDER_VALUE_RE.match(user) or value.startswith(("$", "%", "{", "<")):
            return False
        return _shannon_entropy(value) >= 2.5
    if kind in ("password_assignment", "auth_header", "aws_secret_key"):
        return (
            _shannon_entropy(value) >= 3.5
            and any(c.isdigit() for c in value)
            and any(c.isalpha() for c in value)
        )
    return True


def find_secrets(text: str) -> list[str]:
    found = []
    lowered = text.lower()
    for kind, pattern in _SECRET_PATTERNS.items():
        if kind in _TRIGGERS:
            if not any(t in text for t in _TRIGGERS[kind]) or (kind == "credential_url" and "@" not in text):
                continue
        elif not any(t in lowered for t in _LOWER_TRIGGERS[kind]):
            continue
        matches = pattern.finditer(text)
        if kind == "password_assignment":
            matches = (
                m for m in matches if _PASSWORD_KEY_RE.search(lowered, max(0, m.start() - 24), m.start())
            )
        if any(_looks_like_real_secret(kind, m) for m in matches):
            found.append(kind)
    return found


def find_high_risk_pii(text: str) -> list[str]:
    found = []
    lowered = text.lower()
    if ("ssn" in lowered or "social security" in lowered) and _HIGH_RISK_PII["us_ssn"].search(text):
        found.append("us_ssn")
    # Card numbers are only considered where the document talks about cards at all;
    # grouped digit runs in numeric tables otherwise pass the Luhn check one time in ten.
    if not any(k in lowered for k in _CARD_KEYWORDS):
        return found
    matches = [*_HIGH_RISK_PII["payment_card"].finditer(text), *_CARD_KEYWORD_RE.finditer(lowered)]
    for match in matches:
        digits = re.sub(r"\D", "", match.group(1) if match.re is _CARD_KEYWORD_RE else match.group(0))
        if digits not in _TEST_CARDS and len(set(digits)) > 3 and _luhn_valid(digits):
            found.append("payment_card")
            break
    return found


def _redact_email(match: re.Match[str], counts: Counter[str]) -> str:
    if _SAFE_EMAIL_DOMAINS.search(match.group("domain")) or match.group(0) == EMAIL_PLACEHOLDER:
        return match.group(0)
    counts["email"] += 1
    return EMAIL_PLACEHOLDER


def _redact_ipv4(match: re.Match[str], counts: Counter[str]) -> str:
    value = match.group(0)
    if value in _WELL_KNOWN_IPS:
        return value
    # Four small components are far more often a software version than a host.
    if all(int(part) < 10 for part in value.split(".")):
        return value
    if _VERSION_CONTEXT_RE.search(match.string[max(0, match.start() - 12) : match.start()]):
        return value
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError:
        return value
    if not address.is_global or value.startswith(("0.", "255.")):
        return value
    counts["ipv4"] += 1
    return IPV4_PLACEHOLDER


def _redact_phone(match: re.Match[str], counts: Counter[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    local = digits[-10:]
    if local[3:6] == "555" or len(set(local)) <= 2:
        return match.group(0)
    counts["phone"] += 1
    return PHONE_PLACEHOLDER


def _redact_home_path(match: re.Match[str], counts: Counter[str]) -> str:
    user = match.group("user")
    if user in _GENERIC_USERS or user.lower() in _GENERIC_USERS or user.startswith(("$", "%", "<", "{")):
        return match.group(0)
    counts["home_path_user"] += 1
    return match.group("prefix") + PATH_USER_PLACEHOLDER


def scrub(text: str, *, redact_phones: bool = True) -> PrivacyResult:
    """Reject or redact. ``redact_phones`` is disabled for source code, where
    digit groups that look like phone numbers are almost always data."""
    reasons = [f"privacy.secret.{kind}" for kind in find_secrets(text)]
    reasons += [f"privacy.pii.{kind}" for kind in find_high_risk_pii(text)]
    if reasons:
        return PrivacyResult(text, reasons)
    counts: Counter[str] = Counter()
    if "@" in text:
        text = _EMAIL_RE.sub(lambda m: _redact_email(m, counts), text)
    text = _IPV4_RE.sub(lambda m: _redact_ipv4(m, counts), text)
    if redact_phones:
        text = _PHONE_RE.sub(lambda m: _redact_phone(m, counts), text)
    if "/home/" in text or "Users" in text:
        text = _HOME_PATH_RE.sub(lambda m: _redact_home_path(m, counts), text)
    return PrivacyResult(text, [], counts)

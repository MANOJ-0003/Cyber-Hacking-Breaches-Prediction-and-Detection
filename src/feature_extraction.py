"""Safe, offline URL parsing and a clearly labeled lexical feature prototype.

The repository currently has no source dataset with 87 extracted columns and no
feature specification. The names below describe only the small prototype used
by the original scaffold; they MUST NOT be treated as equivalent to an unknown
87-feature training schema. This module never makes a network request.
"""
from __future__ import annotations

from urllib.parse import urlsplit
import ipaddress
import re

FEATURE_NAMES = [
    "url_length", "hostname_length", "uses_https", "has_ip_host",
    "num_dots", "num_hyphens", "num_at", "num_digits", "num_subdomains",
    "has_suspicious_word", "has_port", "has_path", "has_query",
]
FEATURE_SCHEMA_STATUS = "prototype_only_not_the_expected_87_feature_schema"
SUSPICIOUS_WORDS = ("login", "verify", "account", "secure", "update", "bank", "password", "confirm", "wallet")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def normalize_url(value: str) -> str:
    """Validate and normalize an HTTP(S) URL without opening it.

    A scheme-less domain is interpreted as HTTPS. Other schemes are rejected.
    Credentials are allowed for lexical inspection but are never used to connect.
    """
    if not isinstance(value, str):
        raise ValueError("URL must be text.")
    value = value.strip()
    if not value:
        raise ValueError("Enter a URL to analyze.")
    if len(value) > 2048:
        raise ValueError("URL is too long (maximum 2048 characters).")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("URL contains control characters.")
    if any(char.isspace() for char in value):
        raise ValueError("URL cannot contain whitespace.")

    # Do not reinterpret javascript:, data:, ftp:, or malformed explicit schemes
    # as hostnames by blindly prepending https://.
    if _SCHEME_RE.match(value):
        candidate = value
    else:
        candidate = "https://" + value
    try:
        parsed = urlsplit(candidate)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port  # Access validates malformed and out-of-range ports.
    except ValueError as exc:
        raise ValueError("URL has invalid authority or port syntax.") from exc
    if scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs can be analyzed.")
    if not parsed.netloc or not hostname:
        raise ValueError("Enter a valid HTTP or HTTPS URL with a hostname.")
    if any(char in hostname for char in ("%", "\\", "<", ">", '"', "'")):
        raise ValueError("URL hostname contains invalid characters.")
    if hostname.startswith(".") or ".." in hostname:
        raise ValueError("URL hostname is malformed.")
    # Validate bracketed IPv6 and tolerate normal IPv4/domain/IDN hostnames.
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        try:
            hostname.encode("idna")
        except UnicodeError as exc:
            raise ValueError("URL hostname is malformed.") from exc
    # Keep normalized scheme and input authority/path intact. `port` is read
    # above to force validation; it is otherwise not needed here.
    _ = port
    return candidate


def extract_features(value: str) -> dict[str, int]:
    """Return 13 prototype lexical signals; this is not the missing 87 schema."""
    url = normalize_url(value)
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0
    labels = [label for label in host.split(".") if label]
    searchable = (host + parsed.path + parsed.query).lower()
    return {
        "url_length": len(url),
        "hostname_length": len(host),
        "uses_https": int(parsed.scheme.lower() == "https"),
        "has_ip_host": has_ip,
        "num_dots": host.count("."),
        "num_hyphens": url.count("-"),
        "num_at": url.count("@"),
        "num_digits": sum(ch.isdigit() for ch in url),
        "num_subdomains": max(0, len(labels) - 2),
        "has_suspicious_word": int(any(word in searchable for word in SUSPICIOUS_WORDS)),
        "has_port": int(parsed.port is not None),
        "has_path": int(parsed.path not in ("", "/")),
        "has_query": int(bool(parsed.query)),
    }


def explain_features(features: dict) -> list[str]:
    """Return simple review notes for the prototype signals."""
    notes = []
    if not features["uses_https"]:
        notes.append("The URL uses HTTP rather than HTTPS.")
    if features["has_ip_host"]:
        notes.append("The hostname is an IP address rather than a domain name.")
    if features["has_suspicious_word"]:
        notes.append("The URL contains a word commonly used in account or credential lures.")
    if features["num_subdomains"] >= 3:
        notes.append("The hostname has several subdomain levels.")
    if features["num_hyphens"] >= 3:
        notes.append("The URL contains multiple hyphens.")
    if features["num_at"]:
        notes.append("The URL contains an @ character, which can obscure the destination.")
    if features["url_length"] > 100:
        notes.append("The URL is unusually long.")
    if not notes:
        notes.append("No high-risk lexical indicators in this small rule summary; this does not establish that the site is safe.")
    return notes

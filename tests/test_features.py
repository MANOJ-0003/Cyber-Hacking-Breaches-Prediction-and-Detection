import pytest

from src.feature_extraction import FEATURE_NAMES, extract_features, normalize_url


def test_valid_url_features_follow_declared_order():
    result = extract_features("https://secure.example.com/login?next=home")
    assert list(result) == FEATURE_NAMES
    assert result["uses_https"] == 1
    assert result["has_suspicious_word"] == 1
    assert result["has_query"] == 1


def test_scheme_less_domain_defaults_to_https():
    assert normalize_url("example.com/path") == "https://example.com/path"
    assert extract_features("example.com")["uses_https"] == 1


@pytest.mark.parametrize("value", [
    "", "javascript:alert(1)", "data:text/html,hello", "ftp://example.com",
    "http:///missing-host", "https://example.com:99999/", "https://bad host/",
    "https://example..com/", "https://example.com/\nheader",
])
def test_invalid_urls_raise_clear_validation_error(value):
    with pytest.raises(ValueError):
        normalize_url(value)


def test_non_text_url_is_rejected():
    with pytest.raises(ValueError, match="must be text"):
        normalize_url(None)


def test_unusual_but_parseable_urls_are_inspected_without_network_access():
    result = extract_features("http://user@example.com:8080/a/b?x=1&y=2#fragment")
    assert result["uses_https"] == 0
    assert result["num_at"] == 1
    assert result["has_port"] == 1
    assert result["has_path"] == 1
    assert result["has_query"] == 1


def test_ip_literal_and_ipv6_url_are_supported():
    assert extract_features("http://192.0.2.4/login")["has_ip_host"] == 1
    assert extract_features("https://[2001:db8::1]/")["has_ip_host"] == 1

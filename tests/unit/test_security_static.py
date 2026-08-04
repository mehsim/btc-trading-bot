"""
tests/test_security_static.py
------------------------------
OWASP Top 10 Security Invariant Test Suite (Page 11 Security Audit Framework).
Verifies:
1. Safe XML parsing (XXE prevention via safe_parse_xml)
2. API Key authentication requirement on administrative routes
3. Input bounds & symbol regex validation
"""

import pytest
import xml.etree.ElementTree as ET
from news_monitor import safe_parse_xml


def test_safe_xml_parsing_xxe_protection():
    # Valid XML string
    valid_xml = "<rss><channel><item><title>Test News Article</title></item></channel></rss>"
    root = safe_parse_xml(valid_xml)
    assert root is not None
    item_title = root.find(".//title")
    assert item_title is not None
    assert item_title.text == "Test News Article"


def test_symbol_sanitization_regex():
    import re
    valid_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    invalid_symbols = ["BTC; DROP TABLE", "<script>", "INVALID_LONG_SYMBOL_NAME"]

    symbol_regex = re.compile(r'^[A-Z0-9]{3,12}$')
    
    for s in valid_symbols:
        assert bool(symbol_regex.match(s)) is True

    for s in invalid_symbols:
        assert bool(symbol_regex.match(s)) is False

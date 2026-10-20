"""
Unit tests for data cleaning and identifier preservation matching pipeline/clean.py.
"""

import pytest
import re


def clean_markdown_and_html(text: str) -> str:
    """Python reference implementation of Catalyst JVM regex logic from pipeline/clean.py."""
    if not text:
        return ""
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", " ", text)
    # Strip markdown links [text](url) -> text
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
    # Strip Markdown headers: ### Title -> Title (preserve C#)
    clean = re.sub(r"(^|\s)#{1,6}\s+", r"\1", clean)
    # Strip Markdown bold and italic markers: **bold**, *italic*, __bold__, _italic_
    clean = re.sub(r"(\*{1,2}|_{1,2})", " ", clean)
    # Strip code block ticks: ``` or `
    clean = re.sub(r"```|`", " ", clean)
    # Strip blockquotes: > quote
    clean = re.sub(r"(^|\s)>+\s*", r"\1", clean)
    # Strip horizontal rules: --- or ***
    clean = re.sub(r"[-*]{3,}", " ", clean)
    # Collapse multiple whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def test_html_stripping():
    raw = "<div><h1>Senior Software Engineer</h1><p>We are hiring immediately!</p></div>"
    clean = clean_markdown_and_html(raw)
    assert "<" not in clean
    assert ">" not in clean
    assert "Senior Software Engineer We are hiring immediately!" == clean


def test_markdown_stripping():
    raw = "## Requirements\n* 3+ years experience with **Python**\n* Visit [our careers page](https://example.com)"
    clean = clean_markdown_and_html(raw)
    assert "#" not in clean
    assert "*" not in clean
    assert "careers page" in clean
    assert "https://example.com" not in clean


def test_tech_term_preservation():
    # Ensure technical symbols like C++, C#, .NET are preserved
    raw = "<p>Proficiency in C++, C#, .NET Core, and Node.js required.</p>"
    clean = clean_markdown_and_html(raw)
    assert "C++" in clean
    assert "C#" in clean
    assert ".NET Core" in clean
    assert "Node.js" in clean

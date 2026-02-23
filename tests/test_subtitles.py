"""Tests for backend.utils.subtitles module."""

import pytest
from backend.utils.subtitles import (
    hex_to_ass,
    wrap_text,
    optimize_segments,
    format_timestamp,
    parse_time,
)


class TestHexToAss:
    def test_basic_white(self):
        assert hex_to_ass("#FFFFFF") == "&H00FFFFFF"

    def test_basic_red(self):
        result = hex_to_ass("#FF0000")
        assert result == "&H000000FF"

    def test_basic_green(self):
        result = hex_to_ass("#00FF00")
        assert result == "&H0000FF00"

    def test_basic_blue(self):
        result = hex_to_ass("#0000FF")
        assert result == "&H00FF0000"

    def test_with_alpha(self):
        result = hex_to_ass("#FFFFFF", alpha=0.5)
        # alpha=0.5 => a_val = int(0.5 * 255) = 127 = 0x7F
        assert result == "&H7FFFFFFF"

    def test_shorthand_hex(self):
        result = hex_to_ass("#FFF")
        assert result == "&H00FFFFFF"

    def test_no_hash(self):
        result = hex_to_ass("FF0000")
        assert result == "&H000000FF"

    def test_invalid_color_returns_fallback(self):
        result = hex_to_ass("xyz")
        assert result == "&H00FFFFFF"

    def test_empty_string_returns_fallback(self):
        result = hex_to_ass("")
        assert result == "&H00FFFFFF"


class TestWrapText:
    def test_short_text_unchanged(self):
        assert wrap_text("Hello", max_chars=12) == "Hello"

    def test_long_english_text_wraps(self):
        result = wrap_text("This is a longer sentence", max_chars=12)
        lines = result.split('\n')
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 12

    def test_cjk_text_wraps_by_chars(self):
        result = wrap_text("你好世界測試文字很長的句子", max_chars=4)
        lines = result.split('\n')
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 4

    def test_empty_text(self):
        assert wrap_text("") == ""

    def test_none_text(self):
        assert wrap_text(None) is None

    def test_zero_max_chars(self):
        assert wrap_text("Hello", max_chars=0) == "Hello"

    def test_preserves_existing_newlines(self):
        result = wrap_text("Line1\nLine2", max_chars=20)
        assert "Line1" in result
        assert "Line2" in result


class TestOptimizeSegments:
    def test_empty_input(self):
        assert optimize_segments([]) == []

    def test_single_short_segment(self):
        segments = [{"start": 0, "end": 1, "text": "Hello"}]
        result = optimize_segments(segments, max_chars=14)
        assert len(result) == 1
        assert result[0]["text"] == "Hello"

    def test_filters_empty_segments(self):
        """Empty segments are stripped, and remaining short segments merge."""
        segments = [
            {"start": 0, "end": 1, "text": "Hello"},
            {"start": 1, "end": 2, "text": "   "},
            {"start": 2, "end": 3, "text": "World"},
        ]
        result = optimize_segments(segments, max_chars=14, remove_punctuation=False)
        # "   " is stripped to empty and filtered out.
        # "Hello" (5) + "World" (5) = 10 < 14, so they merge.
        assert len(result) == 1
        assert "Hello" in result[0]["text"]
        assert "World" in result[0]["text"]

    def test_merges_short_segments(self):
        segments = [
            {"start": 0, "end": 0.5, "text": "Hi"},
            {"start": 0.5, "end": 1.0, "text": "there"},
        ]
        result = optimize_segments(segments, max_chars=14, remove_punctuation=False)
        # "Hi" + " " + "there" = 8 chars < 14, should merge
        assert len(result) == 1
        assert "Hi" in result[0]["text"]
        assert "there" in result[0]["text"]

    def test_removes_cjk_punctuation(self):
        segments = [{"start": 0, "end": 1, "text": "你好，世界！"}]
        result = optimize_segments(segments, max_chars=14, remove_punctuation=True)
        assert len(result) == 1
        assert "，" not in result[0]["text"]
        assert "！" not in result[0]["text"]

    def test_preserves_text_with_no_punctuation_removal(self):
        segments = [{"start": 0, "end": 1, "text": "你好，世界！"}]
        result = optimize_segments(segments, max_chars=14, remove_punctuation=False)
        assert result[0]["text"] == "你好，世界！"


class TestFormatTimestamp:
    def test_zero(self):
        result = format_timestamp(0)
        assert result == "00:00,000"

    def test_one_minute(self):
        result = format_timestamp(60)
        assert result == "01:00,000"

    def test_with_milliseconds(self):
        result = format_timestamp(1.5)
        assert result == "00:01,500"

    def test_one_hour(self):
        result = format_timestamp(3600)
        assert result == "01:00:00,000"

    def test_always_include_hours(self):
        result = format_timestamp(30, always_include_hours=True)
        assert result == "00:00:30,000"


class TestParseTime:
    def test_float_passthrough(self):
        assert parse_time(88.5) == 88.5

    def test_int_passthrough(self):
        assert parse_time(60) == 60.0

    def test_string_float(self):
        assert parse_time("88.5") == 88.5

    def test_mm_ss_format(self):
        assert parse_time("1:28") == 88.0

    def test_mm_ss_with_ms(self):
        assert parse_time("1:28.5") == 88.5

    def test_hh_mm_ss_format(self):
        assert parse_time("01:02:03") == 3723.0

    def test_none_returns_zero(self):
        assert parse_time(None) == 0.0

    def test_invalid_string_returns_zero(self):
        assert parse_time("invalid") == 0.0

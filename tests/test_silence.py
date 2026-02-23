"""Tests for backend.utils.silence module."""

from unittest.mock import patch, MagicMock
from backend.utils.silence import detect_silence_ffmpeg


class TestDetectSilenceFfmpeg:
    @patch('backend.utils.silence.subprocess.run')
    def test_parses_silence_segments(self, mock_run):
        """Test that ffmpeg output is correctly parsed into silence segments."""
        mock_result = MagicMock()
        mock_result.stderr = (
            "[silencedetect @ 0x123] silence_start: 1.5\n"
            "[silencedetect @ 0x123] silence_end: 3.0 | silence_duration: 1.5\n"
            "[silencedetect @ 0x123] silence_start: 8.0\n"
            "[silencedetect @ 0x123] silence_end: 10.5 | silence_duration: 2.5\n"
        )
        mock_run.return_value = mock_result

        result = detect_silence_ffmpeg("/fake/path.mp4")

        assert len(result) == 2
        assert result[0] == (1.5, 3.0)
        assert result[1] == (8.0, 10.5)

    @patch('backend.utils.silence.subprocess.run')
    def test_empty_output(self, mock_run):
        """Test handling of no silence detected."""
        mock_result = MagicMock()
        mock_result.stderr = "some other ffmpeg output\nno silence here\n"
        mock_run.return_value = mock_result

        result = detect_silence_ffmpeg("/fake/path.mp4")

        assert result == []

    @patch('backend.utils.silence.subprocess.run')
    def test_unmatched_start_dropped(self, mock_run):
        """Test that an unmatched silence_start at the end is dropped."""
        mock_result = MagicMock()
        mock_result.stderr = (
            "[silencedetect @ 0x123] silence_start: 1.5\n"
            "[silencedetect @ 0x123] silence_end: 3.0 | silence_duration: 1.5\n"
            "[silencedetect @ 0x123] silence_start: 8.0\n"
            # No matching silence_end for 8.0
        )
        mock_run.return_value = mock_result

        result = detect_silence_ffmpeg("/fake/path.mp4")

        assert len(result) == 1
        assert result[0] == (1.5, 3.0)

    @patch('backend.utils.silence.subprocess.run')
    def test_custom_parameters_passed(self, mock_run):
        """Test that custom noise_db and duration are passed to ffmpeg."""
        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        detect_silence_ffmpeg("/fake/path.mp4", noise_db=-40, duration=1.0)

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "-af" in cmd
        af_idx = cmd.index("-af")
        assert "noise=-40dB" in cmd[af_idx + 1]
        assert "d=1.0" in cmd[af_idx + 1]

    @patch('backend.utils.silence.subprocess.run')
    def test_malformed_lines_ignored(self, mock_run):
        """Test that malformed lines are safely ignored."""
        mock_result = MagicMock()
        mock_result.stderr = (
            "silence_start: not_a_number\n"
            "[silencedetect @ 0x123] silence_start: 2.0\n"
            "[silencedetect @ 0x123] silence_end: 4.0 | silence_duration: 2.0\n"
        )
        mock_run.return_value = mock_result

        result = detect_silence_ffmpeg("/fake/path.mp4")

        assert len(result) == 1
        assert result[0] == (2.0, 4.0)

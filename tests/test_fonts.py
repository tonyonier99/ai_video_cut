"""Tests for backend.services.fonts module."""

from unittest.mock import patch, MagicMock
import backend.services.fonts as fonts_module


class TestScanFonts:
    @patch('backend.services.fonts.os.path.exists', return_value=False)
    def test_no_fonts_dir(self, mock_exists):
        """Test that scan_fonts handles missing fonts directory."""
        fonts_module.scan_fonts()
        assert len(fonts_module.FONT_FILE_MAP) == 0

    @patch('backend.services.fonts.os.path.exists', return_value=True)
    @patch('backend.services.fonts.os.listdir', return_value=[])
    def test_empty_fonts_dir(self, mock_listdir, mock_exists):
        """Test that scan_fonts handles empty fonts directory."""
        fonts_module.scan_fonts()
        assert len(fonts_module.FONT_FILE_MAP) == 0

    @patch('backend.services.fonts.os.path.exists', return_value=True)
    @patch('backend.services.fonts.os.listdir', return_value=['test.txt', 'readme.md'])
    def test_non_font_files_ignored(self, mock_listdir, mock_exists):
        """Test that non-font files are ignored."""
        fonts_module.scan_fonts()
        assert len(fonts_module.FONT_FILE_MAP) == 0

    @patch('backend.services.fonts.TTFont')
    @patch('backend.services.fonts.os.path.exists', return_value=True)
    @patch('backend.services.fonts.os.listdir', return_value=['TestFont.ttf'])
    def test_font_with_name_record(self, mock_listdir, mock_exists, mock_ttfont):
        """Test that font family name is extracted from name table."""
        mock_name_record = MagicMock()
        mock_name_record.nameID = 1
        mock_name_record.toUnicode.return_value = "Test Family"

        mock_font = MagicMock()
        mock_font.__getitem__ = MagicMock(return_value=MagicMock(names=[mock_name_record]))
        mock_ttfont.return_value = mock_font

        fonts_module.scan_fonts()

        assert "TestFont" in fonts_module.FONT_FILE_MAP
        assert fonts_module.FONT_FILE_MAP["TestFont"] == "Test Family"

    @patch('backend.services.fonts.TTFont', side_effect=Exception("Parse error"))
    @patch('backend.services.fonts.os.path.exists', return_value=True)
    @patch('backend.services.fonts.os.listdir', return_value=['BrokenFont.otf'])
    def test_broken_font_uses_filename(self, mock_listdir, mock_exists, mock_ttfont):
        """Test that broken font files fall back to using filename as name."""
        fonts_module.scan_fonts()

        assert "BrokenFont" in fonts_module.FONT_FILE_MAP
        assert fonts_module.FONT_FILE_MAP["BrokenFont"] == "BrokenFont"

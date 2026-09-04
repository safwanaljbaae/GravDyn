from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest

from gravdyn.linea_drive_service import (
    list_linea_folder_files,
    download_linea_folder_files,
    _parse_autoindex_links,
)

AUTOINDEX_HTML = """<html>
<head><title>Index of /gravdyn/Data/path/</title></head>
<body>
<h1>Index of /gravdyn/Data/path/</h1><hr><pre><a href="../">../</a>
<a href="file1.dat">file1.dat</a>                                 18-Aug-2026 19:36                 237
<a href="file2.dat">file2.dat</a>                                 18-Aug-2026 19:36                 389
<a href="subdir/">subdir/</a>                                         18-Aug-2026 19:36                   -
</pre><hr></body>
</html>"""


class TestParseAutoindexLinks:
    def test_extracts_files_only(self):
        result = _parse_autoindex_links(AUTOINDEX_HTML)
        assert result == ["file1.dat", "file2.dat"]

    def test_empty_listing(self):
        assert _parse_autoindex_links("<pre><a href=\"../\">../</a></pre>") == []


class TestListLineaFolderFiles:
    @patch("gravdyn.linea_drive_service.requests.get")
    def test_returns_filenames(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text=AUTOINDEX_HTML)
        result = list_linea_folder_files("path")
        assert result == ["file1.dat", "file2.dat"]
        assert mock_get.call_args[0][0] == (
            "https://datasets.linea.org.br/gravdyn/Data/path/"
        )

    @patch("gravdyn.linea_drive_service.requests.get")
    def test_empty_folder_raises(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200, text='<a href="../">../</a>'
        )
        with pytest.raises(FileNotFoundError, match="No files found"):
            list_linea_folder_files("path")

    @patch("gravdyn.linea_drive_service.requests.get")
    def test_404_raises_filenotfound(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        with pytest.raises(FileNotFoundError, match="Folder not found"):
            list_linea_folder_files("path")

    @patch("gravdyn.linea_drive_service.requests.get")
    def test_server_error_raises_runtimeerror(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500, text="boom")
        with pytest.raises(RuntimeError, match="LInEA server error 500"):
            list_linea_folder_files("path")


class TestDownloadLineaFolderFiles:
    @patch("gravdyn.linea_drive_service.requests.get")
    def test_downloads_files(self, mock_get, tmp_path):
        mock_get.side_effect = [
            MagicMock(status_code=200, text=AUTOINDEX_HTML),
            MagicMock(status_code=200, content=b"data1"),
            MagicMock(status_code=200, content=b"data2"),
        ]
        downloaded = download_linea_folder_files("path", output_dir=str(tmp_path))
        assert (tmp_path / "file1.dat").read_bytes() == b"data1"
        assert (tmp_path / "file2.dat").read_bytes() == b"data2"
        assert len(downloaded) == 2

    @patch("gravdyn.linea_drive_service.requests.get")
    def test_overwrites_existing_file(self, mock_get, tmp_path):
        existing = tmp_path / "file1.dat"
        existing.write_text("original")
        mock_get.side_effect = [
            MagicMock(status_code=200, text=AUTOINDEX_HTML),
            MagicMock(status_code=200, content=b"overwritten"),
            MagicMock(status_code=200, content=b"data2"),
        ]
        download_linea_folder_files("path", output_dir=str(tmp_path))
        assert existing.read_bytes() == b"overwritten"

    @patch("gravdyn.linea_drive_service.requests.get")
    def test_creates_output_dir(self, mock_get, tmp_path):
        mock_get.side_effect = [
            MagicMock(status_code=200, text=AUTOINDEX_HTML),
            MagicMock(status_code=200, content=b"data1"),
            MagicMock(status_code=200, content=b"data2"),
        ]
        out_dir = tmp_path / "nested" / "dir"
        download_linea_folder_files("path", output_dir=str(out_dir))
        assert (out_dir / "file1.dat").exists()

    @patch("gravdyn.linea_drive_service.requests.get")
    def test_download_failure_raises(self, mock_get, tmp_path):
        import requests as requests_lib

        mock_get.side_effect = [
            MagicMock(status_code=200, text=AUTOINDEX_HTML),
            requests_lib.exceptions.ConnectionError("no network"),
        ]
        with pytest.raises(RuntimeError, match="Failed to download"):
            download_linea_folder_files("path", output_dir=str(tmp_path))

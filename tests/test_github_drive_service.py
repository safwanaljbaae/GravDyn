from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest

from gravdyn.github_drive_service import (
    list_github_folder_files,
    download_github_folder,
)


class TestListGithubFolderFiles:
    @patch("gravdyn.github_drive_service.requests.get")
    def test_returns_filenames(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {"name": "file1.dat", "type": "file"},
                {"name": "file2.dat", "type": "file"},
                {"name": "subdir", "type": "dir"},
            ],
        )
        result = list_github_folder_files("owner", "repo", "path")
        assert result == ["file1.dat", "file2.dat"]

    @patch("gravdyn.github_drive_service.requests.get")
    def test_empty_folder_raises(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        with pytest.raises(FileNotFoundError, match="No files found"):
            list_github_folder_files("owner", "repo", "path")

    @patch("gravdyn.github_drive_service.requests.get")
    def test_404_raises_filenotfound(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        with pytest.raises(FileNotFoundError, match="Folder not found"):
            list_github_folder_files("owner", "repo", "path")

    @patch("gravdyn.github_drive_service.requests.get")
    def test_no_token_uses_default_headers(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"name": "f.dat", "type": "file"}],
        )
        list_github_folder_files("owner", "repo", "path")
        call_args = mock_get.call_args
        headers = call_args[1].get("headers", {})
        assert "Authorization" not in headers


class TestDownloadGithubFolder:
    @patch("gravdyn.github_drive_service.requests.get")
    def test_downloads_files(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"name": "test.dat", "type": "file",
                            "download_url": "https://example.com/test.dat"}],
            content=b"test data",
        )
        download_github_folder("owner", "repo", "path", output_dir=str(tmp_path))
        assert (tmp_path / "test.dat").exists()
        assert (tmp_path / "test.dat").read_bytes() == b"test data"

    @patch("gravdyn.github_drive_service.requests.get")
    def test_overwrites_existing_file(self, mock_get, tmp_path):
        existing = tmp_path / "existing.dat"
        existing.write_text("original")
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"name": "existing.dat", "type": "file",
                            "download_url": "https://example.com/existing.dat"}],
            content=b"overwritten",
        )
        download_github_folder("owner", "repo", "path", output_dir=str(tmp_path))
        assert existing.read_text() == "overwritten"

    @patch("gravdyn.github_drive_service.requests.get")
    def test_creates_output_dir(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"name": "test.dat", "type": "file",
                            "download_url": "https://example.com/test.dat"}],
            content=b"data",
        )
        out_dir = tmp_path / "nested" / "dir"
        download_github_folder("owner", "repo", "path", output_dir=str(out_dir))
        assert (out_dir / "test.dat").exists()

import unittest
from unittest.mock import patch, MagicMock
from datasets.utils import download_file_if_not_exists_or_size_diff

class TestDownloadFileIfNotExistsOrSizeDiff(unittest.TestCase):

    @patch('datasets.utils.file_is_downloaded_with_size_same')
    @patch('datasets.utils.download_file')
    def test_file_not_exists(self, mock_download_file, mock_file_is_downloaded_with_size_same):
        # Mock file_is_downloaded_with_size_same to return False
        mock_file_is_downloaded_with_size_same.side_effect = [False, True]

        # Call the function
        download_file_if_not_exists_or_size_diff('http://example.com/file', '/path/to/file')

        # Assert download_file was called
        mock_download_file.assert_called_once_with('http://example.com/file', '/path/to/file')

    @patch('datasets.utils.file_is_downloaded_with_size_same')
    @patch('datasets.utils.download_file')
    def test_file_exists_with_different_size(self, mock_download_file, mock_file_is_downloaded_with_size_same):
        # Mock file_is_downloaded_with_size_same to return False
        mock_file_is_downloaded_with_size_same.side_effect = [False, True]

        # Call the function
        download_file_if_not_exists_or_size_diff('http://example.com/file', '/path/to/file')

        # Assert download_file was called
        mock_download_file.assert_called_once_with('http://example.com/file', '/path/to/file')

    @patch('datasets.utils.file_is_downloaded_with_size_same')
    @patch('datasets.utils.download_file')
    def test_file_exists_with_same_size(self, mock_download_file, mock_file_is_downloaded_with_size_same):
        # Mock file_is_downloaded_with_size_same to return True
        mock_file_is_downloaded_with_size_same.return_value = True

        # Call the function
        download_file_if_not_exists_or_size_diff('http://example.com/file', '/path/to/file')

        # Assert download_file was not called
        mock_download_file.assert_not_called()

if __name__ == '__main__':
    unittest.main()
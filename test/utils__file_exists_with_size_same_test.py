import unittest
from unittest.mock import patch, mock_open

from datasets.utils import file_is_downloaded_with_size_same

class TestFileIsDownloadedWithSizeSame(unittest.TestCase):

    @patch('datasets.utils.is_file_downloaded')
    @patch('datasets.utils.is_file_size_same')
    def test_file_exists_and_size_matches(self, mock_is_file_downloaded, mock_is_file_size_same):
        mock_is_file_downloaded.return_value = True
        mock_is_file_size_same.return_value = True
        
        result = file_is_downloaded_with_size_same('http://example.com/file', '/path/to/file')
        self.assertTrue(result)

    @patch('datasets.utils.is_file_downloaded')
    @patch('datasets.utils.is_file_size_same')
    def test_file_exists_and_size_does_not_match(self, mock_is_file_downloaded, mock_is_file_size_same):
        mock_is_file_downloaded.return_value = True
        mock_is_file_size_same.return_value = False
        
        result = file_is_downloaded_with_size_same('http://example.com/file', '/path/to/file')
        self.assertFalse(result)

    @patch('datasets.utils.is_file_downloaded')
    @patch('datasets.utils.is_file_size_same')
    def test_file_does_not_exist(self, mock_is_file_downloaded, mock_is_file_size_same):
        mock_is_file_downloaded.return_value = False
        mock_is_file_size_same.return_value = True
        
        result = file_is_downloaded_with_size_same('http://example.com/file', '/path/to/file')
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
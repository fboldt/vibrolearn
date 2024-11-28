import unittest
import os
from unittest.mock import patch
from datasets.utils import is_file_downloaded

class TestIsFileDownloaded(unittest.TestCase):

    @patch('os.path.isfile')
    def test_is_file_downloaded_file_exists(self, mock_isfile):
        url = "http://example.com/file.txt"
        folder_path = "/some/folder"
        mock_isfile.return_value = True

        result = is_file_downloaded(url, folder_path)
        self.assertTrue(result)
        mock_isfile.assert_called_once_with(os.path.join(folder_path, "file.txt"))

    @patch('os.path.isfile')
    def test_is_file_downloaded_file_not_exists(self, mock_isfile):
        url = "http://example.com/file.txt"
        folder_path = "/some/folder"
        mock_isfile.return_value = False

        result = is_file_downloaded(url, folder_path)
        self.assertFalse(result)
        mock_isfile.assert_called_once_with(os.path.join(folder_path, "file.txt"))

if __name__ == '__main__':
    unittest.main()

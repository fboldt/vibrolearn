import unittest
from unittest.mock import patch, mock_open

import requests
from datasets.utils import download_file

class TestDownloadFile(unittest.TestCase):

    @patch('datasets.utils.requests.get')
    def test_download_file_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b'Test content'
        
        with patch('builtins.open', mock_open()) as mocked_file:
            download_file('http://example.com/file', '/path/to/destination')
            mocked_file.assert_called_once_with('/path/to/destination/file', 'wb')
            mocked_file().write.assert_called_once_with(b'Test content')

    @patch('datasets.utils.requests.get')
    def test_download_file_http_error(self, mock_get):
        mock_get.return_value.status_code = 404
        result = download_file('http://example.com/file', '/path/to/destination')
        self.assertFalse(result)

    @patch('datasets.utils.requests.get')
    def test_download_file_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError
        
        with self.assertRaises(Exception):
            download_file('http://example.com/file', '/path/to/destination')

if __name__ == '__main__':
    unittest.main()
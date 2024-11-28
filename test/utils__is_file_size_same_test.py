import os
import unittest
from unittest.mock import patch, mock_open
from datasets.utils import is_file_size_same

class MockHeader():
    def __init__(self, ContentLength=1024):
        self.headers = {'Content-Length': ContentLength}
    def get(self, key, _):
        return self.headers[key]
    
class MockHeadResponse():
    def __init__(self, status_code=200, headers=MockHeader(1024)):
        self.status_code = status_code
        self.headers = headers


class TestIsFileSizeSame(unittest.TestCase):

    @patch('datasets.utils.requests.head')
    @patch('datasets.utils.os.path.getsize')
    @patch('datasets.utils.os.path.isfile')
    def test_file_size_same(self, mock_isfile, mock_getsize, mock_head):
        mock_isfile.return_value = True
        mock_getsize.return_value = 1024
        mock_head.return_value = MockHeadResponse()
        url = 'http://example.com/file'
        file_path = '/path/to/file'
        result = is_file_size_same(url, file_path)
        self.assertTrue(result)

    @patch('datasets.utils.requests.head')
    @patch('datasets.utils.os.path.getsize')
    @patch('datasets.utils.os.path.isfile')
    def test_file_size_different(self, mock_isfile, mock_getsize, mock_head):
        mock_isfile.return_value = True
        mock_getsize.return_value = 1024
        mock_head.return_value = MockHeadResponse(200, MockHeader(2048))
        url = 'http://example.com/file'
        file_path = '/path/to/file'
        result = is_file_size_same(url, file_path)
        self.assertFalse(result)

    @patch('datasets.utils.requests.head')
    @patch('datasets.utils.os.path.getsize')
    @patch('datasets.utils.os.path.isfile')
    def test_file_not_found(self, mock_isfile, mock_getsize, mock_head):
        mock_isfile.return_value = True
        mock_getsize.return_value = 1024
        mock_head.return_value = MockHeadResponse(404)
        url = 'http://example.com/file'
        file_path = '/path/to/file'
        result = is_file_size_same(url, file_path)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
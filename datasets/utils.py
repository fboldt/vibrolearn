import os
import requests

import urllib.parse

def is_file_downloaded(url, folder_path):
    # Parse the URL to get the file name
    parsed_url = urllib.parse.urlparse(url)
    file_name = os.path.basename(parsed_url.path)
    
    # Check if the file exists in the specified folder
    file_path = os.path.join(folder_path, file_name)
    return os.path.isfile(file_path)

def is_file_size_same(url, file_path):    
    # Check if the file exists
    if not os.path.isfile(file_path):
        return False
    
    # Get the size of the local file
    local_file_size = os.path.getsize(file_path)
    
    # Get the size of the file from the URL
    response = requests.head(url)
    if response.status_code != 200:
        return False
    url_file_size = int(response.headers.get('Content-Length', 0))
    
    return local_file_size == url_file_size
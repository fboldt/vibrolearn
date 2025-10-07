import os
import requests
import urllib.parse
import csv
import scipy.io
import numpy as np


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


def read_registers_from_config(config_path):
    registers = []
    with open(config_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row = {k.strip(): v.strip() if v is not None else v for k, v in row.items()}
            registers.append(row)
    return registers


def filter_registers_by_key_value_sequence(registers, key_value_sequence):
    return [reg for reg in registers if all(reg.get(k) in v for k, v in key_value_sequence)]


def get_values_by_key(registers, key):
    return set([reg.get(key) for reg in registers if key in reg])


def get_all_keys_and_values(registers):
    for key in registers[0].keys():
        if key == 'filename':
            continue
        values = get_values_by_key(registers, key)
        print(f"{key}: {values}")


def load_matlab_file(file_path):
    mat = scipy.io.loadmat(file_path)
    return mat


def get_matlab_acquisition(mat, source):
    variable_name = None
    for key in mat.keys():
        if source in key:
            variable_name = key
            break
    if variable_name is not None:
        return mat[variable_name]
    else:
        raise KeyError(f"Variable '{variable_name}' not found in the MATLAB file.")


def load_acquisition(register, raw_dir_path, channel):
    filename = register['filename']
    file_path =  f"{raw_dir_path}/{filename}"
    mat = load_matlab_file(file_path)
    return get_matlab_acquisition(mat, channel)


def split_acquisition(acquisition, segment_length):
    num_segments = acquisition.shape[0] // segment_length
    segments = np.empty((num_segments, segment_length, acquisition.shape[1]), dtype=acquisition.dtype)
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segments[i] = acquisition[start:end, :]
    return segments


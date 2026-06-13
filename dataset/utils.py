import os
import requests
import urllib.parse
import csv
import scipy.io
import numpy as np
import rarfile #need install unrar


def download_file_from_register(raw_dir_path, register):
    os.makedirs(raw_dir_path, exist_ok=True)
    file_url = register['url'].strip()
    acquisition_file = register['acquisition_file'].strip()
    if not is_acquisition_file_downloaded(acquisition_file, raw_dir_path):
        download_with_retries(file_url, raw_dir_path)
        if file_url.lower().endswith('.rar'):
            extract_rar(file_url, raw_dir_path)


def is_acquisition_file_downloaded(acquisition_file, folder_path):
    # Check if the acquisition file exists in the specified folder
    file_path = os.path.join(folder_path, acquisition_file)
    return os.path.isfile(file_path)


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


def extract_rar(file_url, raw_dir_path):
    filename = file_url.split('/')[-1]
    file_path = os.path.join(raw_dir_path, filename)
    with rarfile.RarFile(file_path) as rf:
        rf.extractall(raw_dir_path)


def download_with_retries(file_url, raw_dir_path, max_trials=5):
    filename = file_url.split('/')[-1]
    file_path = os.path.join(raw_dir_path, filename)
    trials_left = max_trials
    while (not is_file_downloaded(file_url, raw_dir_path) or 
           not is_file_size_same(file_url, file_path)) and trials_left > 0:
        print(f"Downloading file {file_path} (Trials left: {trials_left})...")
        download_from_url(file_url, file_path)
        trials_left -= 1
    if trials_left == 0:
        raise Exception(f"Failed to download file {file_path} correctly after multiple attempts.")


def download_from_url(url, file_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        print("Trying again...")


def download_dataset(config_path, raw_dir_path, filenames=None):
    registers = read_registers_from_config(config_path)
    if filenames:
        registers = [r for r in registers if (r['url']).split('/')[-1].strip() in filenames]
    for register in registers:
        download_file_from_register(raw_dir_path, register)
    print(f"Dataset downloaded to {raw_dir_path}")


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


def filter_registers_by_key_value_absence(registers, key_value_sequence):
    return [reg for reg in registers if all(reg.get(k) not in v for k, v in key_value_sequence)]


def get_values_by_key(registers, key):
    return set([reg.get(key) for reg in registers if key in reg])


def get_all_keys_and_values(registers):
    for key in registers[0].keys():
        if key == 'acquisition_file' or key == 'url':
            continue
        values = get_values_by_key(registers, key)
        print(f"{key}: {values}")


def load_matlab_file(file_path):
    try:
        mat = scipy.io.loadmat(file_path)
        return mat
    except Exception as e:
        print(f"Error loading MATLAB file {file_path}: {e}")
        raise e


def load_matlab_acquisition(file_path, channels):
    try:
        mat = load_matlab_file(file_path)
    except Exception as e:
        raise e
    acquisition = []
    for channel in channels:
        for key in mat.keys():
            if channel in key:
                acquisition.append(mat[key])
                break
    if acquisition:
        return np.concatenate(acquisition, axis=1)
    else:
        raise KeyError(f"Variables '{channels}' not found in the MATLAB file.")


def split_acquisition(acquisition, segment_length):
    num_segments = acquisition.shape[0] // segment_length
    segments = np.empty((num_segments, segment_length, acquisition.shape[1]), dtype=acquisition.dtype)
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segments[i] = acquisition[start:end, :]
    return segments


def target_array(value, length):
    target_type = type(value)
    if target_type == str:
        target_type = np.dtype('U' + str(len(value)))
    return np.full(length, value, dtype=target_type)


def get_X_y(registers, raw_dir_path, channels_columns, segment_length, load_acquisition_func):
    X_list = []
    y_list = []
    if len(registers) == 0:
        return np.empty((0, segment_length, 1)), np.empty((0,), dtype='U10')
    for key_value in channels_columns.keys():
        key, value = key_value.split(":")
        value = eval(value)
        filtered_registers = filter_registers_by_key_value_sequence(registers, [[key, value]])
        actual_channel_columns = channels_columns[key_value]
        for register in filtered_registers:
            segments, targets = extract_segments_and_targets(raw_dir_path, actual_channel_columns, segment_length, load_acquisition_func, register)
            X_list.append(segments)
            y_list.append(targets)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

import numpy as np


def get_X_y_domains(
    registers,
    raw_dir_path,
    channels_columns,
    segment_length,
    load_acquisition_func,
    domain_key="severity"
):
    X_list = []
    y_list = []
    domains_list = []

    if len(registers) == 0:
        return (
            np.empty((0, segment_length, 1)),
            np.empty((0,), dtype="U10"),
            np.empty((0,), dtype="U10")
        )

    for key_value in channels_columns.keys():
        key, value = key_value.split(":")
        value = eval(value)

        filtered_registers = filter_registers_by_key_value_sequence(
            registers,
            [[key, value]]
        )

        actual_channel_columns = channels_columns[key_value]

        for register in filtered_registers:
            segments, targets = extract_segments_and_targets(
                raw_dir_path,
                actual_channel_columns,
                segment_length,
                load_acquisition_func,
                register
            )

            domain_value = register[domain_key]

            domains = np.full(
                shape=(len(targets),),
                fill_value=domain_value
            )

            X_list.append(segments)
            y_list.append(targets)
            domains_list.append(domains)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    domains = np.concatenate(domains_list, axis=0)

    return X, y, domains


def extract_segments_and_targets(raw_dir_path, channels_columns, segment_length, load_acquisition_func, register):
    acquisition = get_acquisition_data(raw_dir_path, channels_columns, load_acquisition_func, register)
    segments, targets = prepare_segments_and_targets(segment_length, register, acquisition)
    return segments,targets


def prepare_segments_and_targets(segment_length, register, acquisition):
    segments = split_acquisition(acquisition, segment_length=segment_length)
    targets = target_array(register['condition'], segments.shape[0])
    return segments,targets


def get_acquisition_data(raw_dir_path, channels_columns, load_acquisition_func, register):
    file_path = os.path.join(raw_dir_path, register['acquisition_file'])
    channels = get_channels_from_register(channels_columns, register)
    try:
        acquisition = load_acquisition_func(file_path, channels=channels)
    except Exception as e:
        download_file_from_register(raw_dir_path, register)
        acquisition = load_acquisition_func(file_path, channels=channels)
    return acquisition


def get_channels_from_register(channels_columns, register):
    channels = []
    for channel_column in channels_columns:
        if channel_column not in register:
            raise KeyError(f"Channel column '{channel_column}' not found in register.")
        channels.append(register[channel_column])
    return channels


def get_fold(fold_filters, config_file):
    registers = read_registers_from_config(config_file)
    fold = []
    for fold_filter in fold_filters:
        filtered = filter_registers_by_key_value_sequence(
            registers, 
            [[k, v] for k, v in fold_filter.items()])
        fold.extend(filtered)
    return fold


def get_folds(experimental_setup, combination_key, config_file):
    folds = {}
    for fold_key in experimental_setup["setup"][combination_key]:
        fold = get_fold(experimental_setup["setup"][combination_key][fold_key], config_file=config_file)
        folds[fold_key] = fold
    return folds


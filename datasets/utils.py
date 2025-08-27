import os
import requests
import urllib.parse
import csv
from pathlib import Path
import pprint

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

if __name__ == "__main__":
    config_file = Path(__file__).parent / "cwru/config.csv"
    registers = read_registers_from_config(config_file)
    filtered_registers = filter_registers_by_key_value_sequence(registers, [('sample_rate', ['12000']), ('faulty_bearing', ['None', 'Drive End']), ('load', ['0']), ('condition', ['Normal'])])
    pprint.pprint(filtered_registers)
    print(f"Filtered registers: {len(filtered_registers)}")
    for key in filtered_registers[0].keys():
        if key == 'filename':
            continue
        values = get_values_by_key(filtered_registers, key)
        print(f"{key}: {values}")


from pathlib import Path
from pprint import pprint
from datasets.utils import filter_registers_by_key_value_sequence, read_registers_from_config, load_acquisition, split_acquisition
from datasets.cwru.utils import get_raw_dir_path, get_code_from_faulty_bearing


if __name__ == "__main__":
    config_file = Path(__file__).parent / "datasets/cwru/config.csv"
    registers = read_registers_from_config(config_file)
    filtered_registers = filter_registers_by_key_value_sequence(registers, [('sample_rate', ['12000']), ('faulty_bearing', ['None', 'Drive End', 'Fan End']), ('load', ['0']), ('condition', ['Normal', 'Inner Race'])])
    pprint(filtered_registers)
    print(f"Filtered registers: {len(filtered_registers)}")
    # get_all_keys_and_values(registers)
    first_register = filtered_registers[0]
    acquisition = load_acquisition(first_register, 
                                   get_raw_dir_path(), 
                                   channel=get_code_from_faulty_bearing(first_register['faulty_bearing']))
    print(f"Acquisition shape: {acquisition.shape}")
    segments = split_acquisition(acquisition, segment_length=2048)
    print(f"Segments shape: {segments.shape}")

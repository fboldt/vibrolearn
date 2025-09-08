from pathlib import Path
from pprint import pprint
from datasets.cwru.utils import load_acquisition
from datasets.utils import filter_registers_by_key_value_sequence, read_registers_from_config


if __name__ == "__main__":
    config_file = Path(__file__).parent / "datasets/cwru/config.csv"
    registers = read_registers_from_config(config_file)
    filtered_registers = filter_registers_by_key_value_sequence(registers, [('sample_rate', ['12000']), ('faulty_bearing', ['None', 'Drive End', 'Fan End']), ('load', ['0']), ('condition', ['Normal', 'Inner Race'])])
    pprint(filtered_registers)
    print(f"Filtered registers: {len(filtered_registers)}")
    # get_all_keys_and_values(registers)
    acquisition = load_acquisition(filtered_registers[0])
    print(f"Acquisition shape: {acquisition.shape}")
    

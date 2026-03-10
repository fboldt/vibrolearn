from datasets.utils import download_file_from_register, read_registers_from_config

# Testa CWRU
cwru_registers = read_registers_from_config("datasets/cwru/config.csv")
download_file_from_register("raw_data/cwru", cwru_registers[0])

# Testa Paderborn (K001.rar)
paderborn_registers = read_registers_from_config("datasets/paderborn/config.csv")
download_file_from_register("raw_data/paderborn", paderborn_registers[0])
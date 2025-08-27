from pathlib import Path
from datasets.utils import read_registers_from_config, filter_registers_by_key_value_sequence
import torch
from torch.utils.data import Dataset

config_file = read_registers_from_config(Path(__file__).parent / "config.csv")
registers = read_registers_from_config(config_file)
severity07 = filter_registers_by_key_value_sequence(registers, [('sample_rate', ['12000']), ('severity', ['0.007'])])
severity14 = filter_registers_by_key_value_sequence(registers, [('sample_rate', ['12000']), ('severity', ['0.014'])])
severity21 = filter_registers_by_key_value_sequence(registers, [('sample_rate', ['12000']), ('severity', ['0.021'])])
severity28 = filter_registers_by_key_value_sequence(registers, [('sample_rate', ['12000']), ('severity', ['0.028'])])

class CWRUDataset(Dataset):
    def __init__(self, registers):
        self.registers = registers

    def __len__(self):
        return len(self.registers)

    def __getitem__(self, idx):
        register = self.registers[idx]
        # Example: assuming register has 'data' and 'label' keys
        data = torch.tensor(register['data'], dtype=torch.float32)
        label = torch.tensor(register['label'], dtype=torch.long)
        return data, label

fold0 = CWRUDataset(severity07)
fold1 = CWRUDataset(severity14)
fold2 = CWRUDataset(severity21)
fold3 = CWRUDataset(severity28)

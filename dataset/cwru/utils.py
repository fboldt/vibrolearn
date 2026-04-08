from dataset.utils import filter_registers_by_key_value_sequence, read_registers_from_config


config_file = "dataset/cwru/config.csv"


def get_folds(combination):
    registers = read_registers_from_config(config_file)
    folds = []
    for item in combination:
        filtered = filter_registers_by_key_value_sequence(
            registers, 
            [[k, v] for k, v in item.items()])
        folds.extend(filtered)
    return folds


def get_list_of_folds(combinations, comb_index):
    folds = []
    for combination in combinations[comb_index]:
        fold = get_folds(combination)
        folds.append(fold)
    return folds


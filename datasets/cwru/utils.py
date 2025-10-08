from pathlib import Path
from datasets.utils import filter_registers_by_key_value_sequence, read_registers_from_config


def get_code_from_faulty_bearing(faulty_bearing):
    if faulty_bearing == 'Drive End':
        return 'DE'
    elif faulty_bearing == 'Fan End':
        return 'FE'
    else:
        return 'DE'


def get_raw_dir_path():
    return "raw_data/cwru"


def get_fold_rauber_loca_et_al(normal_load, fault_bearing_severity, faulty_bearing, sample_rate='12000'):
    config_file = Path(__file__).parent / "config.csv"
    registers = read_registers_from_config(config_file)
    normal = filter_registers_by_key_value_sequence(registers, [('sample_rate', [sample_rate]), ('faulty_bearing', ['None']), ('load', [str(normal_load)]), ('condition', ['Normal'])])
    faulty = filter_registers_by_key_value_sequence(registers, [('sample_rate', [sample_rate]), ('faulty_bearing', [faulty_bearing]), ('severity', [f"{fault_bearing_severity:.3f}"])])
    fold = []
    fold.extend(normal)
    fold.extend(faulty)
    return fold


def get_list_of_folds_rauber_loca_et_al(faulty_bearing, sample_rate='12000'):
    fold0 = get_fold_rauber_loca_et_al(normal_load=0, fault_bearing_severity=0.007, faulty_bearing=faulty_bearing, sample_rate=sample_rate)
    fold1 = get_fold_rauber_loca_et_al(normal_load=1, fault_bearing_severity=0.014, faulty_bearing=faulty_bearing, sample_rate=sample_rate)
    fold2 = get_fold_rauber_loca_et_al(normal_load=2, fault_bearing_severity=0.021, faulty_bearing=faulty_bearing, sample_rate=sample_rate)
    fold3 = get_fold_rauber_loca_et_al(normal_load=3, fault_bearing_severity=0.028, faulty_bearing=faulty_bearing, sample_rate=sample_rate)
    folds = [fold0, fold1, fold2, fold3]
    return folds

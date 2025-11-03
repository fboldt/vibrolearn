from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pathlib import Path
from dataset.utils import filter_registers_by_key_value_sequence, read_registers_from_config, filter_registers_by_key_value_absence, get_list_of_X_y
from assesment.crossvalidation import performance

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
faulty_bearing = ['Drive End', 'Fan End']
sample_rate = '12000'
segment_length = 2048
channels_columns = ["DE", "FE"]

rauber_loca_et_al_combination_rounds = [
    [(0, 0.007), (1, 0.014), (2, 0.021), (3, 0.028)],
    [(3, 0.007), (0, 0.014), (1, 0.021), (2, 0.028)],
    [(2, 0.007), (3, 0.014), (0, 0.021), (1, 0.028)],
    [(1, 0.007), (2, 0.014), (3, 0.021), (0, 0.028)],
    [(0, 0.014), (1, 0.007), (2, 0.021), (3, 0.028)],
    [(3, 0.014), (0, 0.007), (1, 0.021), (2, 0.028)],
    [(2, 0.014), (3, 0.007), (0, 0.021), (1, 0.028)],
    [(1, 0.014), (2, 0.007), (3, 0.021), (0, 0.028)],
]

def get_fold_rauber_loca_et_al(normal_load, fault_bearing_severity, faulty_bearing, sample_rate='12000'):
    config_file = "dataset/cwru/config.csv"
    registers = read_registers_from_config(config_file)
    normal = filter_registers_by_key_value_sequence(registers, [('sample_rate', [sample_rate]), ('faulty_bearing', ['None']), ('load', [str(normal_load)]), ('condition', ['Normal'])])
    faulty = filter_registers_by_key_value_sequence(registers, [('sample_rate', [sample_rate]), ('faulty_bearing', faulty_bearing), ('severity', [f"{fault_bearing_severity:.3f}"])])
    fold = []
    fold.extend(normal)
    fold.extend(faulty)
    return fold


def get_list_of_folds_rauber_loca_et_al(faulty_bearing, sample_rate='12000', combination=0):
    folds = []
    for normal_load, fault_bearing_severity in rauber_loca_et_al_combination_rounds[combination%len(rauber_loca_et_al_combination_rounds)]:
        fold = get_fold_rauber_loca_et_al(normal_load=normal_load, fault_bearing_severity=fault_bearing_severity, faulty_bearing=faulty_bearing, sample_rate=sample_rate)
        folds.append(fold)
    return folds


def perform_fold_combination(model, combination, verbose=False):
    list_of_folds = get_list_of_folds_rauber_loca_et_al(faulty_bearing=faulty_bearing, sample_rate=sample_rate, combination=combination)
    filter_folds_by_channel_absence(list_of_folds)
    list_of_folds = filter_folds_by_length(list_of_folds, min_length=2)
    list_of_X_y = get_list_of_X_y(list_of_folds, raw_dir_path="raw_data/cwru", channels_columns=channels_columns, segment_length=segment_length)
    scores_per_fold = performance(model, list_of_X_y, list_of_metrics, verbose=verbose)
    return scores_per_fold

def filter_folds_by_length(list_of_folds, min_length=2):
    list_of_folds = [fold for fold in list_of_folds if len(fold) >= min_length]
    return list_of_folds

def filter_folds_by_channel_absence(list_of_folds):
    for f in range(len(list_of_folds)):
        for channel_column in channels_columns:
            list_of_folds[f] = filter_registers_by_key_value_absence(list_of_folds[f], [(channel_column, ['None'])])

def run(model, verbose=False):
    combination = 0
    scores_per_fold = perform_fold_combination(model, combination, verbose=verbose)
    return scores_per_fold

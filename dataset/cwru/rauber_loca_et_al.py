from dataset.utils import filter_registers_by_key_value_sequence, get_list_of_X_y, load_matlab_acquisition, merge_X_y_from_lists, read_registers_from_config
from assesment.crossvalidation import performance

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

def get_fold(normal_load, fault_bearing_severity, faulty_bearing, sample_rate):
    config_file = "dataset/cwru/config.csv"
    registers = read_registers_from_config(config_file)
    normal = filter_registers_by_key_value_sequence(
        registers, 
        [('sample_rate', [sample_rate]), 
         ('faulty_bearing', ['None']), 
         ('load', [str(normal_load)]), 
         ('condition', ['Normal'])])
    faulty = filter_registers_by_key_value_sequence(
        registers, 
        [('sample_rate', [sample_rate]), 
         ('faulty_bearing', faulty_bearing), 
         ('severity', [f"{fault_bearing_severity:.3f}"])])
    fold = []
    fold.extend(normal)
    fold.extend(faulty)
    return fold

def get_list_of_folds(faulty_bearing, sample_rate, combination):
    folds = []
    for normal_load, fault_bearing_severity in rauber_loca_et_al_combination_rounds[combination%len(rauber_loca_et_al_combination_rounds)]:
        fold = get_fold(normal_load=normal_load, fault_bearing_severity=fault_bearing_severity, faulty_bearing=faulty_bearing, sample_rate=sample_rate)
        if len(fold) > 0:
            folds.append(fold)
    return folds

def single_channel_X_y(combination, segment_length, sample_rate, faulty_bearing, channel_column):
    list_of_folds = get_list_of_folds([faulty_bearing], sample_rate, combination)
    return get_list_of_X_y(list_of_folds, raw_dir_path="raw_data/cwru", channels_columns=[channel_column], segment_length=segment_length, load_acquisition_func=load_matlab_acquisition)

def single_channel_X_y_DE_FE_12k(combination, segment_length):
    sample_rate = '12000'
    X_y_DE = single_channel_X_y(combination, segment_length, sample_rate, 'Drive End', 'DE')
    X_y_FE = single_channel_X_y(combination, segment_length, sample_rate, 'Fan End', 'FE')
    list_of_X_y = merge_X_y_from_lists(X_y_DE, X_y_FE)
    return list_of_X_y

def run_experiment(model, list_of_metrics, debug=False):
    combinations = range(len(rauber_loca_et_al_combination_rounds))
    verbose=False
    if debug:
        combinations = [0]
        verbose = True
    segment_length = 2048
    scores = None
    for combination in combinations:
        list_of_X_y = single_channel_X_y_DE_FE_12k(combination, segment_length)
        scores = performance(model, list_of_X_y, list_of_metrics=list_of_metrics, verbose=verbose)
    return scores

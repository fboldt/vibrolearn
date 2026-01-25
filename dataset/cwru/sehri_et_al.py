from assesment.crossvalidation import performance as crossvalidation
from assesment.traintest import performance as holdout
from dataset.utils import filter_registers_by_key_value_sequence, get_list_of_X_y, load_matlab_acquisition, read_registers_from_config
from pprint import pprint

#               ### training -------------------------###   ### testing --###
papers_split = [(['0', '1', '2', '3'], ['0.007', '0.014']), (['0'], ['0.021'])]

def get_papers_split(loads, severities):
    sample_rate = "48000"
    config_file = "dataset/cwru/config.csv"
    prlzs = ['None', '6']
    registers = read_registers_from_config(config_file)
    filtered = filter_registers_by_key_value_sequence(
        registers,
        [('sample_rate', [sample_rate]),
         ('load', loads), 
         ('severity', severities),
         ('prlz', prlzs)]
    )
    return filtered

def get_list_of_papers_splits():
    train_test_split = []
    for loads, severities in papers_split:
        fold = get_papers_split(loads, severities)
        train_test_split.append(fold)
    return train_test_split

def run_papers_experiment(model, list_of_metrics):
    segment_length = 1024
    list_of_folds = get_list_of_papers_splits()
    # pprint(list_of_folds)
    list_of_X_y = get_list_of_X_y(
        list_of_folds,
        raw_dir_path="raw_data/cwru",
        channels_columns=['DE'],
        segment_length=segment_length,
        load_acquisition_func=load_matlab_acquisition
    )
    scores = holdout(model, list_of_X_y[0][0], list_of_X_y[0][1], list_of_X_y[1][0], list_of_X_y[1][1], list_of_metrics=list_of_metrics)    
    return scores

proposed_cross_validation_folds = ['0.007', '0.014', '0.021']

def get_proposed_cross_validation_fold(severity):
    sample_rate = "48000"
    config_file = "dataset/cwru/config.csv"
    prlzs = ['None', '6']
    registers = read_registers_from_config(config_file)
    filtered = filter_registers_by_key_value_sequence(
        registers,
        [('sample_rate', [sample_rate]),
         ('severity', [f"{severity:.3f}"]),
         ('prlz', prlzs)]
    )
    return filtered

def get_list_of_proposed_cross_validation_folds():
    folds = []
    for severity in proposed_cross_validation_folds:
        fold = get_proposed_cross_validation_fold(float(severity))
        folds.append(fold)
    return folds

def run_proposed_experiment(model, list_of_metrics, debug=False):
    segment_length = 1024
    list_of_folds = get_list_of_proposed_cross_validation_folds()
    pprint(list_of_folds)
    if debug:
        list_of_folds = [list_of_folds[0]]
    list_of_X_y = get_list_of_X_y(
        list_of_folds,
        raw_dir_path="raw_data/cwru",
        channels_columns=['DE'],
        segment_length=segment_length,
        load_acquisition_func=load_matlab_acquisition
    )
    scores = crossvalidation(model, list_of_X_y, list_of_metrics=list_of_metrics, verbose=debug)
    return scores
                
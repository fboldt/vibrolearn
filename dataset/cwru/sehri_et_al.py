import pprint
from assesment.crossvalidation import performance as crossvalidation
from assesment.crossvalidation import get_average_scores
from assesment.traintest import print_scores
from assesment.traintest import performance as holdout
from dataset.utils import filter_registers_by_key_value_sequence, get_list_of_X_y, get_train_test_split, load_matlab_acquisition, read_registers_from_config

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
    list_of_X_y = get_list_of_X_y(
        list_of_folds,
        raw_dir_path="raw_data/cwru",
        channels_columns=['DE'],
        segment_length=segment_length,
        load_acquisition_func=load_matlab_acquisition
    )
    scores = holdout(model, list_of_X_y[0][0], list_of_X_y[0][1], list_of_X_y[1][0], list_of_X_y[1][1], list_of_metrics=list_of_metrics) 
    return scores

papers_inspired_cross_validation_folds = ['0.007', '0.014', '0.021']

def get_papers_inspired_cross_validation_fold(severity):
    sample_rate = "48000"
    config_file = "dataset/cwru/config.csv"
    prlzs = ['None', '6']
    registers = read_registers_from_config(config_file)
    filtered = filter_registers_by_key_value_sequence(
        registers,
        [('sample_rate', [sample_rate]),
         ('severity', [severity]),
         ('prlz', prlzs)]
    )
    return filtered

def get_list_of_papers_inspired_cross_validation_folds():
    folds = []
    for severity in papers_inspired_cross_validation_folds:
        fold = get_papers_inspired_cross_validation_fold(severity)
        folds.append(fold)
    return folds

def run_papers_inspired_experiment(model, list_of_metrics):
    segment_length = 1024
    list_of_folds = get_list_of_papers_inspired_cross_validation_folds()
    list_of_X_y = get_list_of_X_y(
        list_of_folds,
        raw_dir_path="raw_data/cwru",
        channels_columns=['DE'],
        segment_length=segment_length,
        load_acquisition_func=load_matlab_acquisition
    )
    scores = crossvalidation(model, list_of_X_y, list_of_metrics=list_of_metrics, verbose=True)
    return scores

proposed_cross_validation_combinations = [
    [
        [('Inner Race', '0.007'),('Outer Race', '0.014'),('Ball', '0.021')],
        [('Inner Race', '0.014'),('Outer Race', '0.021'),('Ball', '0.007')],
        [('Inner Race', '0.021'),('Outer Race', '0.007'),('Ball', '0.014')]
    ],
    [
        [('Inner Race', '0.007'),('Outer Race', '0.021'),('Ball', '0.014')],
        [('Inner Race', '0.014'),('Outer Race', '0.007'),('Ball', '0.021')],
        [('Inner Race', '0.021'),('Outer Race', '0.014'),('Ball', '0.007')]
    ]
]

def get_fold(combination):
    sample_rate = "48000"
    config_file = "dataset/cwru/config.csv"
    registers = read_registers_from_config(config_file)
    prlzs = ['None', '6']
    fold = []
    for faulty_bearing, fault_bearing_severity in combination:
        filtered = filter_registers_by_key_value_sequence(
            registers, 
            [('sample_rate', [sample_rate]), 
            ('condition', [faulty_bearing]), 
            ('severity', [fault_bearing_severity]),
            ('prlz', prlzs)])
        fold.extend(filtered)
    return fold

def get_list_of_proposed_cross_validation_folds(comb_index=0):
    folds = []
    for combination in proposed_cross_validation_combinations[comb_index]:
        fold = get_fold(combination)
        folds.append(fold)
    return folds

def run_proposed_experiment(model, list_of_metrics):
    segment_length = 1024
    list_of_scores = []
    for comb_index in range(len(proposed_cross_validation_combinations)):
        print("Combination index:", comb_index)
        scores = perform_cross_validation(model, list_of_metrics, segment_length, comb_index)
        list_of_scores.extend(scores)
    print("Average scores over all combinations:")
    print_scores(get_average_scores(list_of_scores))
    return list_of_scores

def perform_cross_validation(model, list_of_metrics, segment_length, comb_index):
    list_of_folds = get_list_of_proposed_cross_validation_folds(comb_index=comb_index)
    list_of_X_y = get_list_of_X_y(
            list_of_folds,
            raw_dir_path="raw_data/cwru",
            channels_columns=['DE'],
            segment_length=segment_length,
            load_acquisition_func=load_matlab_acquisition
        )
    scores = crossvalidation(model, list_of_X_y, list_of_metrics=list_of_metrics, verbose=True)
    return scores

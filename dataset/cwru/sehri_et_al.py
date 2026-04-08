from dataset.cwru.dataloader import get_sehri_et_all_X_y
from utils.assesment import holdout
from dataset.utils import filter_registers_by_key_value_sequence, read_registers_from_config

#               ### training -------------------------###   ### testing --###
papers_split = [(['0', '1', '2', '3'], ['0.007', '0.014']), (['0'], ['0.021'])]
# papers_split = [
#         [('Inner Race', '0.007', ['0', '1', '2', '3']),
#          ('Outer Race', '0.007', ['0', '1', '2', '3']),
#          ('Ball',       '0.007', ['0', '1', '2', '3']),
#          ('Inner Race', '0.014', ['0', '1', '2', '3']),
#          ('Outer Race', '0.014', ['0', '1', '2', '3']),
#          ('Ball',       '0.014', ['0', '1', '2', '3'])],
#         [('Inner Race', '0.021', ['0']),
#          ('Outer Race', '0.021', ['0']),
#          ('Ball',       '0.021', ['0'])]
# ]

proposed_cross_validation_combinations = [
    [
        [('Inner Race', '0.007', ['0', '1', '2', '3']),
         ('Outer Race', '0.007', ['0', '1', '2', '3']),
         ('Ball',       '0.007', ['0', '1', '2', '3'])],
        [('Inner Race', '0.014', ['0', '1', '2', '3']),
         ('Outer Race', '0.014', ['0', '1', '2', '3']),
         ('Ball',       '0.014', ['0', '1', '2', '3'])],
        [('Inner Race', '0.021', ['0', '1', '2', '3']),
         ('Outer Race', '0.021', ['0', '1', '2', '3']),
         ('Ball',       '0.021', ['0', '1', '2', '3'])]
    ],
    [
        [('Inner Race', '0.007', ['0', '1', '2', '3']),
         ('Outer Race', '0.014', ['0', '1', '2', '3']),
         ('Ball',       '0.021', ['0', '1', '2', '3'])],
        [('Inner Race', '0.014', ['0', '1', '2', '3']),
         ('Outer Race', '0.021', ['0', '1', '2', '3']),
         ('Ball',       '0.007', ['0', '1', '2', '3'])],
        [('Inner Race', '0.021', ['0', '1', '2', '3']),
         ('Outer Race', '0.007', ['0', '1', '2', '3']),
         ('Ball',       '0.014', ['0', '1', '2', '3'])]
    ],
    [
        [('Inner Race', '0.007', ['0', '1', '2', '3']),
         ('Outer Race', '0.021', ['0', '1', '2', '3']),
         ('Ball',       '0.014', ['0', '1', '2', '3'])],
        [('Inner Race', '0.014', ['0', '1', '2', '3']),
         ('Outer Race', '0.007', ['0', '1', '2', '3']),
         ('Ball',       '0.021', ['0', '1', '2', '3'])],
        [('Inner Race', '0.021', ['0', '1', '2', '3']),
         ('Outer Race', '0.014', ['0', '1', '2', '3']),
         ('Ball',       '0.007', ['0', '1', '2', '3'])]
    ]
]

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

def get_folds(combination):
    sample_rate = "48000"
    config_file = "dataset/cwru/config.csv"
    registers = read_registers_from_config(config_file)
    prlzs = ['None', '6']
    folds = []
    for faulty_bearing, fault_bearing_severity, loads in combination:
        filtered = filter_registers_by_key_value_sequence(
            registers, 
            [('sample_rate', [sample_rate]), 
            ('condition', [faulty_bearing]), 
            ('severity', [fault_bearing_severity]),
            ('load', loads),
            ('prlz', prlzs)])
        folds.extend(filtered)
    return folds

def get_list_of_papers_splits():
    train_test_split = []
    for loads, severities in papers_split:
        fold = get_papers_split(loads, severities)
        train_test_split.append(fold)
    return train_test_split

def run_papers_experiment(model, list_of_metrics):
    list_of_folds = get_list_of_papers_splits()
    model.set_load_function(get_sehri_et_all_X_y)
    scores = holdout(model, list_of_folds, test_fold_index=1, list_of_metrics=list_of_metrics) 
    return scores

def get_list_of_folds(comb_index=0):
    folds = []
    for combination in proposed_cross_validation_combinations[comb_index]:
        fold = get_folds(combination)
        folds.append(fold)
    return folds

def run_papers_inspired_experiment(model, list_of_metrics):
    list_of_folds = get_list_of_folds(comb_index=0)
    model.set_load_function(get_sehri_et_all_X_y)
    scores = []
    for fold_idx in range(len(list_of_folds)):
        fold_scores = holdout(model, list_of_folds, test_fold_index=fold_idx, list_of_metrics=list_of_metrics)
        scores.append(fold_scores)
    return scores

def run_proposed_experiment(model, list_of_metrics):
    list_of_scores = []
    for comb_index in range(len(proposed_cross_validation_combinations)):
        list_of_folds = get_list_of_folds(comb_index=comb_index)
        model.set_load_function(get_sehri_et_all_X_y)
        for fold_idx in range(len(list_of_folds)):
            fold_scores = holdout(model, list_of_folds, test_fold_index=fold_idx, list_of_metrics=list_of_metrics)
            list_of_scores.append(fold_scores)
    return list_of_scores

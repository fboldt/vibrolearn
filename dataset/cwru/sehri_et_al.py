from dataset.cwru.dataloader import get_sehri_et_al_X_y
from utils.assesment import holdout
from dataset.utils import get_folds


papers_setup = {
    "holdout": { 
        "training": [ 
            {   "condition":   ['Inner Race', 'Outer Race', 'Ball'], 
                "severity":    ['0.007', '0.014'], 
                "sample_rate": ["48000"], "prlz": ['None', '6']}
        ],
        "testing": [
            {   "condition":   ['Inner Race', 'Outer Race', 'Ball'], 
                "severity":    ['0.021'], "load": ['0'], 
                "sample_rate": ["48000"], "prlz": ['None', '6']}
        ],
    }
}


proposed_setup = {
    "cross_validation_0": {
        "fold_0_0": [ 
            { "condition": ['Inner Race', 'Outer Race', 'Ball'], 
              "severity":  ['0.007'], "sample_rate": ["48000"], 
              "prlz": ['None', '6']}
        ],
        "fold_0_1": [
            { "condition": ['Inner Race', 'Outer Race', 'Ball'], 
              "severity":  ['0.014'], "sample_rate": ["48000"], 
              "prlz": ['None', '6']}
        ],
        "fold_0_2": [ 
            { "condition": ['Inner Race', 'Outer Race', 'Ball'], 
              "severity":  ['0.021'], "sample_rate": ["48000"], 
              "prlz": ['None', '6']}
        ]
    },
    "cross_validation_1": {
        "fold_1_0": [ 
            { "condition": ['Inner Race'], "severity": ['0.007'], "sample_rate": ["48000"], },
            { "condition": ['Outer Race'], "severity": ['0.014'], "sample_rate": ["48000"], "prlz": ['6']},
            { "condition": ['Ball'],       "severity": ['0.021'], "sample_rate": ["48000"], },
        ],
        "fold_1_1": [ 
            { "condition": ['Inner Race'], "severity": ['0.014'], "sample_rate": ["48000"], },
            { "condition": ['Outer Race'], "severity": ['0.021'], "sample_rate": ["48000"], "prlz": ['6']},
            { "condition": ['Ball'],       "severity": ['0.007'], "sample_rate": ["48000"], },
        ],
        "fold_1_2": [ 
            { "condition": ['Inner Race'], "severity": ['0.021'], "sample_rate": ["48000"], },
            { "condition": ['Outer Race'], "severity": ['0.007'], "sample_rate": ["48000"], "prlz": ['6']},
            { "condition": ['Ball'],       "severity": ['0.014'], "sample_rate": ["48000"], },
        ],
    },
    "cross_validation_2": {
        "fold_2_0": [ 
            { "condition": ['Inner Race'], "severity": ['0.007'], "sample_rate": ["48000"], },
            { "condition": ['Outer Race'], "severity": ['0.021'], "sample_rate": ["48000"], "prlz": ['6']},
            { "condition": ['Ball'],       "severity": ['0.014'], "sample_rate": ["48000"], },
        ],
        "fold_2_1": [ 
            { "condition": ['Inner Race'], "severity": ['0.014'], "sample_rate": ["48000"], },
            { "condition": ['Outer Race'], "severity": ['0.007'], "sample_rate": ["48000"], "prlz": ['6']},
            { "condition": ['Ball'],       "severity": ['0.021'], "sample_rate": ["48000"], },
        ],
        "fold_2_2": [ 
            { "condition": ['Inner Race'], "severity": ['0.021'], "sample_rate": ["48000"], },
            { "condition": ['Outer Race'], "severity": ['0.014'], "sample_rate": ["48000"], "prlz": ['6']},
            { "condition": ['Ball'],       "severity": ['0.007'], "sample_rate": ["48000"], },
        ],
    }
}


config_file = "dataset/cwru/config.csv"


def run_papers_experiment(model, list_of_metrics):
    list_of_folds = get_folds(papers_setup, "holdout", config_file)
    model.set_load_function(get_sehri_et_al_X_y)
    scores = holdout(model, list_of_folds, test_fold_key="testing", list_of_metrics=list_of_metrics)
    return scores


def run_papers_inspired_experiment(model, list_of_metrics):
    folds = get_folds(proposed_setup, "cross_validation_0", config_file)
    model.set_load_function(get_sehri_et_al_X_y)
    scores = {}
    for fold_key in folds:
        fold_scores = holdout(model, folds, test_fold_key=fold_key, list_of_metrics=list_of_metrics)
        scores[fold_key] = fold_scores
    return scores


def run_proposed_experiment(model, list_of_metrics):
    scores = {}
    model.set_load_function(get_sehri_et_al_X_y)
    for key in proposed_setup:
        folds = get_folds(proposed_setup, key, config_file)
        for fold_key in folds:
            fold_scores = holdout(model, folds, test_fold_key=fold_key, list_of_metrics=list_of_metrics)
            scores[fold_key] = fold_scores
    return scores

from dataset.cwru.dataloader import get_sehri_et_al_X_y
from utils.assesment import holdout
from dataset.cwru.utils import get_list_of_folds


papers_split = [
    [ # unique combination
        [ # training fold
            {   "condition":   ['Inner Race', 'Outer Race', 'Ball'], 
                "severity":    ['0.007', '0.014'], 
                "sample_rate": ["48000"], "prlz": ['None', '6']}
        ],
        [ # testing fold
            {   "condition":   ['Inner Race', 'Outer Race', 'Ball'], 
                "severity":    ['0.021'], "load": ['0'], 
                "sample_rate": ["48000"], "prlz": ['None', '6']}
        ],
    ]
]


proposed_combinations = [
    [ # combination 0
        [ # fold 0
            { "condition": ['Inner Race', 'Outer Race', 'Ball'], 
              "severity":  ['0.007'], "sample_rate": ["48000"], 
              "prlz": ['None', '6']}
        ],
        [ # fold 1
            { "condition": ['Inner Race', 'Outer Race', 'Ball'], 
              "severity":  ['0.014'], "sample_rate": ["48000"], 
              "prlz": ['None', '6']}
        ],
        [ # fold 2
            { "condition": ['Inner Race', 'Outer Race', 'Ball'], 
              "severity":  ['0.021'], "sample_rate": ["48000"], 
              "prlz": ['None', '6']}
        ]
    ],
    [ # combination 1
        [ # fold 4
            { "condition": ['Inner Race'], "severity": ['0.007'], "sample_rate": ["48000"], },
            { "condition": ['Outer Race'], "severity": ['0.014'], "sample_rate": ["48000"], "prlz": ['6']},
            { "condition": ['Ball'],       "severity": ['0.021'], "sample_rate": ["48000"], },
        ],
        [ # fold 5
            { "condition": ['Inner Race'], "severity": ['0.014'], "sample_rate": ["48000"], },
            { "condition": ['Outer Race'], "severity": ['0.021'], "sample_rate": ["48000"], "prlz": ['6']},
            { "condition": ['Ball'],       "severity": ['0.007'], "sample_rate": ["48000"], },
        ],
        [ # fold 6
            { "condition": ['Inner Race'], "severity": ['0.021'], "sample_rate": ["48000"], },
            { "condition": ['Outer Race'], "severity": ['0.007'], "sample_rate": ["48000"], "prlz": ['6']},
            { "condition": ['Ball'],       "severity": ['0.014'], "sample_rate": ["48000"], },
        ],
    ],
    [ # combination 2
        [ # fold 7
            { "condition": ['Inner Race'], "severity": ['0.007'], "sample_rate": ["48000"], },
            { "condition": ['Outer Race'], "severity": ['0.021'], "sample_rate": ["48000"], "prlz": ['6']},
            { "condition": ['Ball'],       "severity": ['0.014'], "sample_rate": ["48000"], },
        ],
        [ # fold 8
            { "condition": ['Inner Race'], "severity": ['0.014'], "sample_rate": ["48000"], },
            { "condition": ['Outer Race'], "severity": ['0.007'], "sample_rate": ["48000"], "prlz": ['6']},
            { "condition": ['Ball'],       "severity": ['0.021'], "sample_rate": ["48000"], },
        ],
        [ # fold 9
            { "condition": ['Inner Race'], "severity": ['0.021'], "sample_rate": ["48000"], },
            { "condition": ['Outer Race'], "severity": ['0.014'], "sample_rate": ["48000"], "prlz": ['6']},
            { "condition": ['Ball'],       "severity": ['0.007'], "sample_rate": ["48000"], },
        ],
    ]
]


def run_papers_experiment(model, list_of_metrics):
    list_of_folds = get_list_of_folds(papers_split, comb_index=0)
    model.set_load_function(get_sehri_et_al_X_y)
    scores = holdout(model, list_of_folds, test_fold_index=1, list_of_metrics=list_of_metrics) 
    return scores


def run_papers_inspired_experiment(model, list_of_metrics):
    list_of_folds = get_list_of_folds(proposed_combinations, comb_index=0)
    model.set_load_function(get_sehri_et_al_X_y)
    scores = []
    for fold_idx in range(len(list_of_folds)):
        fold_scores = holdout(model, list_of_folds, test_fold_index=fold_idx, list_of_metrics=list_of_metrics)
        scores.append(fold_scores)
    return scores


def run_proposed_experiment(model, list_of_metrics):
    list_of_scores = []
    model.set_load_function(get_sehri_et_al_X_y)
    for comb_index in range(len(proposed_combinations)):
        list_of_folds = get_list_of_folds(proposed_combinations, comb_index=comb_index)
        for fold_idx in range(len(list_of_folds)):
            fold_scores = holdout(model, list_of_folds, test_fold_index=fold_idx, list_of_metrics=list_of_metrics)
            list_of_scores.append(fold_scores)
    return list_of_scores

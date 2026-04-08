from dataset.cwru.dataloader import get_rauber_loca_et_al_X_y
from dataset.cwru.utils import get_list_of_folds
from utils.assesment import holdout


rauber_loca_et_al_combinations = [
    [ #combination 0  
        [ #fold 0
            { "condition": ['Normal'], "load": ['0'], "sample_rate": ["12000"], },
            { "severity": ['0.007'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 1
            { "condition": ['Normal'], "load": ['1'], "sample_rate": ["12000"], },
            { "severity": ['0.014'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 2
            { "condition": ['Normal'], "load": ['2'], "sample_rate": ["12000"], },
            { "severity": ['0.021'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 3
            { "condition": ['Normal'], "load": ['3'], "sample_rate": ["12000"], },
            { "severity": ['0.028'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
    ],
    [ #combination 1
        [ #fold 0
            { "condition": ['Normal'], "load": ['3'], "sample_rate": ["12000"], },
            { "severity": ['0.007'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 1
            { "condition": ['Normal'], "load": ['0'], "sample_rate": ["12000"], },
            { "severity": ['0.014'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 2
            { "condition": ['Normal'], "load": ['1'], "sample_rate": ["12000"], },
            { "severity": ['0.021'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 3
            { "condition": ['Normal'], "load": ['2'], "sample_rate": ["12000"], },
            { "severity": ['0.028'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
    ],
    [ #combination 2
        [ #fold 0
            { "condition": ['Normal'], "load": ['2'], "sample_rate": ["12000"], },
            { "severity": ['0.007'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 1
            { "condition": ['Normal'], "load": ['3'], "sample_rate": ["12000"], },
            { "severity": ['0.014'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 2
            { "condition": ['Normal'], "load": ['0'], "sample_rate": ["12000"], },
            { "severity": ['0.021'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 3
            { "condition": ['Normal'], "load": ['1'], "sample_rate": ["12000"], },
            { "severity": ['0.028'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
    ],
    [ #combination 3
        [ #fold 0
            { "condition": ['Normal'], "load": ['1'], "sample_rate": ["12000"], },
            { "severity": ['0.007'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 1
            { "condition": ['Normal'], "load": ['2'], "sample_rate": ["12000"], },
            { "severity": ['0.014'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 2
            { "condition": ['Normal'], "load": ['3'], "sample_rate": ["12000"], },
            { "severity": ['0.021'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 3
            { "condition": ['Normal'], "load": ['0'], "sample_rate": ["12000"], },
            { "severity": ['0.028'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
    ],
    [ #combination 4
        [ #fold 0
            { "condition": ['Normal'], "load": ['0'], "sample_rate": ["12000"], },
            { "severity": ['0.014'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 1
            { "condition": ['Normal'], "load": ['1'], "sample_rate": ["12000"], },
            { "severity": ['0.007'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 2
            { "condition": ['Normal'], "load": ['2'], "sample_rate": ["12000"], },
            { "severity": ['0.021'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 3
            { "condition": ['Normal'], "load": ['3'], "sample_rate": ["12000"], },
            { "severity": ['0.028'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
    ],
    [ #combination 5
        [ #fold 0
            { "condition": ['Normal'], "load": ['3'], "sample_rate": ["12000"], },
            { "severity": ['0.014'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 1
            { "condition": ['Normal'], "load": ['0'], "sample_rate": ["12000"], },
            { "severity": ['0.007'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 2
            { "condition": ['Normal'], "load": ['1'], "sample_rate": ["12000"], },
            { "severity": ['0.021'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 3
            { "condition": ['Normal'], "load": ['2'], "sample_rate": ["12000"], },
            { "severity": ['0.028'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
    ],
    [ #combination 6
        [ #fold 0
            { "condition": ['Normal'], "load": ['2'], "sample_rate": ["12000"], },
            { "severity": ['0.014'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 1
            { "condition": ['Normal'], "load": ['3'], "sample_rate": ["12000"], },
            { "severity": ['0.007'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 2
            { "condition": ['Normal'], "load": ['0'], "sample_rate": ["12000"], },
            { "severity": ['0.021'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 3
            { "condition": ['Normal'], "load": ['1'], "sample_rate": ["12000"], },
            { "severity": ['0.028'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
    ],
    [ #combination 7
        [ #fold 0
            { "condition": ['Normal'], "load": ['1'], "sample_rate": ["12000"], },
            { "severity": ['0.014'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 1
            { "condition": ['Normal'], "load": ['2'], "sample_rate": ["12000"], },
            { "severity": ['0.007'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 2
            { "condition": ['Normal'], "load": ['3'], "sample_rate": ["12000"], },
            { "severity": ['0.021'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
        [ #fold 3
            { "condition": ['Normal'], "load": ['0'], "sample_rate": ["12000"], },
            { "severity": ['0.028'], "faulty_bearing": ["Drive End"], "sample_rate": ["12000"], },
        ],
    ],
]


def run_experiment(model, list_of_metrics):
    print("Running rauber_loca_et_al experiment")
    list_of_scores = []
    model.set_load_function(get_rauber_loca_et_al_X_y)
    for combination in range(len(rauber_loca_et_al_combinations)):
        print(f"Combination {combination}")
        list_of_folds = get_list_of_folds(rauber_loca_et_al_combinations, comb_index=combination)
        for fold_idx in range(len(list_of_folds)):
            print(f"Fold {fold_idx}")
            fold_scores = holdout(model, list_of_folds, test_fold_index=fold_idx, list_of_metrics=list_of_metrics)
            list_of_scores.append(fold_scores)
    return list_of_scores

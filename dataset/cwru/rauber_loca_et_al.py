from dataset.cwru.dataloader import get_rauber_loca_et_al_X_y
from dataset.cwru.utils import get_list_of_folds
from utils.assesment import holdout


rauber_loca_et_al_combinations = [
    [ #combination 0
        [(['Normal'], ['0'], ['0']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.007'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['1']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.014'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['2']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.021'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['3']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.028'], ['0', '1', '2', '3'])]
    ],
    [ #combination 1
        [(['Normal'], ['0'], ['3']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.007'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['0']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.014'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['1']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.021'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['2']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.028'], ['0', '1', '2', '3'])]
    ],
    [ #combination 2
        [(['Normal'], ['0'], ['2']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.007'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['3']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.014'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['0']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.021'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['1']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.028'], ['0', '1', '2', '3'])]
    ],
    [ #combination 3
        [(['Normal'], ['0'], ['0']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.007'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['1']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.014'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['2']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.021'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['3']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.028'], ['0', '1', '2', '3'])]
    ],
    [ #combination 4
        [(['Normal'], ['0'], ['0']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.014'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['1']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.007'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['2']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.021'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['3']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.028'], ['0', '1', '2', '3'])]
    ],
    [ #combination 5
        [(['Normal'], ['0'], ['3']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.014'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['0']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.007'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['1']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.021'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['2']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.028'], ['0', '1', '2', '3'])]
    ],
    [ #combination 6
        [(['Normal'], ['0'], ['2']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.014'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['3']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.007'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['0']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.021'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['1']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.028'], ['0', '1', '2', '3'])]
    ],
    [ #combination 7
        [(['Normal'], ['0'], ['1']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.014'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['2']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.007'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['3']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.021'], ['0', '1', '2', '3'])],
        [(['Normal'], ['0'], ['0']),
         (['Inner Race', 'Outer Race', 'Ball'], ['0.028'], ['0', '1', '2', '3'])]
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

from datasets.cwru.utils import get_code_from_faulty_bearing, get_list_of_folds_rauber_loca_et_al
from datasets.utils import get_X_y, get_train_test_split

def test_experiments():
    faulty_bearing='Drive End'
    sample_rate='12000'
    folds = get_list_of_folds_rauber_loca_et_al(faulty_bearing=faulty_bearing, sample_rate=sample_rate)
    channel = get_code_from_faulty_bearing(faulty_bearing)
    list_of_X_y = []
    for fold in folds:
        print(f"Number of acquisitions: {len(fold)}")
        X, y = get_X_y(fold, raw_dir_path="raw_data/cwru", channel=channel)
        list_of_X_y.append((X, y))
    for i in range(len(folds)):
        print(f"{'#'*3} Fold {i}:")
        X_train, y_train, X_test, y_test = get_train_test_split(list_of_X_y, test_fold_index=i)
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

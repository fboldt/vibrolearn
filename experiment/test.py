from sklearn.metrics import accuracy_score
from assesment.train_test import performance
from dataset.cwru.utils import get_code_from_faulty_bearing, get_list_of_folds_rauber_loca_et_al
from dataset.utils import get_list_of_X_y, get_train_test_split
from sklearn.ensemble import RandomForestClassifier

def run():
    faulty_bearing='Drive End'
    sample_rate='12000'
    folds = get_list_of_folds_rauber_loca_et_al(faulty_bearing=faulty_bearing, sample_rate=sample_rate)
    channel = get_code_from_faulty_bearing(faulty_bearing)
    list_of_X_y = get_list_of_X_y(folds, raw_dir_path="raw_data/cwru", channel=channel, segment_length=2048)
    i = 0
    print(f"{'#'*3} Fold {i}:")
    X_train, y_train, X_test, y_test = get_train_test_split(list_of_X_y, test_fold_index=i)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    scores = performance(RandomForestClassifier(n_estimators=100, random_state=42), X_train.reshape(X_train.shape[0], -1), y_train, X_test.reshape(X_test.shape[0], -1), y_test, list_metrics=[accuracy_score])
    print(f"Scores: {scores}")



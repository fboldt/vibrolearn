from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from assesment.crossvalidation import performance, print_scores_per_fold
from dataset.cwru.utils import get_code_from_faulty_bearing, get_list_of_folds_rauber_loca_et_al
from dataset.utils import get_list_of_X_y
from estimators.randomforest import model 


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def run():
    faulty_bearing='Drive End'
    sample_rate='12000'
    folds = get_list_of_folds_rauber_loca_et_al(faulty_bearing=faulty_bearing, sample_rate=sample_rate)
    channel = get_code_from_faulty_bearing(faulty_bearing)
    list_of_X_y = get_list_of_X_y(folds, raw_dir_path="raw_data/cwru", channel=channel, segment_length=2048)

    list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
    scores_per_fold = performance(model, list_of_X_y, list_metrics=list_of_metrics)
    print_scores_per_fold(scores_per_fold)
    
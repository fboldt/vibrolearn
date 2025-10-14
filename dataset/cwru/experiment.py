
from assesment.crossvalidation import performance
from dataset.cwru.utils import get_code_from_faulty_bearing, get_list_of_folds_rauber_loca_et_al
from dataset.utils import get_list_of_X_y


def rauber_loca_et_al(model, list_of_metrics):
    faulty_bearing='Drive End'
    sample_rate='12000'
    folds = get_list_of_folds_rauber_loca_et_al(faulty_bearing=faulty_bearing, sample_rate=sample_rate)
    channel = get_code_from_faulty_bearing(faulty_bearing)
    list_of_X_y = get_list_of_X_y(folds, raw_dir_path="raw_data/cwru", channel=channel, segment_length=2048)
    scores_per_fold = performance(model, list_of_X_y, list_metrics=list_of_metrics)
    return scores_per_fold

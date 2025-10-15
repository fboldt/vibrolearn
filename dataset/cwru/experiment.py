
from assesment.crossvalidation import performance
from dataset.cwru.utils import get_code_from_faulty_bearing, get_list_of_folds_rauber_loca_et_al
from dataset.utils import get_list_of_X_y, merge_X_y_from_lists


def rauber_loca_et_al(model, list_of_metrics, verbose=False):
    sample_rate='12000'

    def get_loxy_rlea(faulty_bearing):
        lofde = get_list_of_folds_rauber_loca_et_al(faulty_bearing=faulty_bearing, sample_rate=sample_rate)
        channel = get_code_from_faulty_bearing(faulty_bearing)
        loxy = get_list_of_X_y(lofde, raw_dir_path="raw_data/cwru", channel=channel, segment_length=2048)
        return loxy

    loxyde = get_loxy_rlea(faulty_bearing='Drive End')
    loxyfe = get_loxy_rlea(faulty_bearing='Fan End')
    loxy = merge_X_y_from_lists([loxyde, loxyfe])
    scores_per_fold = performance(model, loxy, list_metrics=list_of_metrics, verbose=verbose)
    return scores_per_fold


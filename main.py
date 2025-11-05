from dataset.utils import read_registers_from_config
from estimators import rfwfe, adaboost, randomforest
from experiment.cwru_rauber_loca_et_al import perform_fold_combination_for_all_channels_available
from pprint import pprint

def run(model, combination, verbose=False):
    scores_per_fold = perform_fold_combination_for_all_channels_available(model, combination, verbose=verbose)
    return scores_per_fold


if __name__ == "__main__":
    # print("Running Random Forest with Raw Features...")
    # result  = run(randomforest.model, 0, verbose=True)

    # print("\nRunning Random Forest with Heterogeneous Features...")
    # result  = run(rfwfe.model, 0, verbose=True)

    # print("\nRunning AdaBoost with Heterogeneous Features...")
    # result = run(adaboost.model, 0, verbose=True)

    registers = read_registers_from_config("dataset/cwru/config.csv")
    pprint(registers[:4])    

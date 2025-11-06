from dataset.cwru.rauber_loca_et_al import single_channel_X_y_DE_FE_12k
from estimators import rfwfe, adaboost, randomforest
from assesment.crossvalidation import performance
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pprint import pprint


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]


def run(model, verbose=False):
    combination = 0
    segment_length = 2048
    list_of_X_y = single_channel_X_y_DE_FE_12k(combination, segment_length)
    scores = performance(model, list_of_X_y, list_of_metrics=list_of_metrics, verbose=verbose)
    return scores



if __name__ == "__main__":
    # print("Running Random Forest with Raw Features...")
    # result  = run(randomforest.model, verbose=True)

    print("\nRunning Random Forest with Heterogeneous Features...")
    result  = run(rfwfe.model, verbose=True)

    # print("\nRunning AdaBoost with Heterogeneous Features...")
    # result = run(adaboost.model, verbose=True)

    # registers = read_registers_from_config("dataset/cwru/config.csv")
    # pprint(registers[:4])    

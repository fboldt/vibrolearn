from dataset.cwru.rauber_loca_et_al import single_channel_X_y_DE_FE_12k
from assesment.crossvalidation import performance
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from estimator.RandomForest import RandomForest 


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
    result  = run(RandomForest(), verbose=True)

    # registers = read_registers_from_config("dataset/cwru/config.csv")[:1]
    # # pprint(registers[:4])    
    # X, y = get_X_y(registers, "raw_data/cwru/", ["DE"], 2048, load_matlab_acquisition)
    # print(f"X shape: {X.shape}, y shape: {y.shape}")
    # festextractor = WaveletPackage(wavelet='db4', mode='symmetric', maxlevel=4)
    # features = festextractor.transform(X)
    # print(f"Extracted features shape: {features.shape}")

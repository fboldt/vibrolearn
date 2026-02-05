from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from feature.extraction import WaveletFeatures

model = make_pipeline(WaveletFeatures(), RandomForestClassifier())

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]

if __name__ == "__main__":
    #'''
    from dataset.cwru.sehri_et_al import run_papers_experiment
    scores = run_papers_experiment(model, list_of_metrics)
    print("Scores for papers experiment:")
    pprint(scores)
    ''' 
    from dataset.cwru.sehri_et_al import run_papers_inspired_experiment
    from assesment.nested import performance
    run_papers_inspired_experiment(model, list_of_metrics, performance)
    #'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from pprint import pprint
from feature.extraction import WaveletPackage

def get_wavelet_random_forest():
    n_estimators=300
    random_state=42
    n_jobs=-1
    wavelet='db4'
    mode='symmetric'
    maxlevel=4
    model = Pipeline(
        steps=[
            (
                "wavelet",
                WaveletPackage(
                    wavelet=wavelet,
                    mode=mode,
                    maxlevel=maxlevel,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
    return model

def f1_macro(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='macro')

def print_dict_of_scores(scores):
    print(20 * "-")
    for metric_name, score in scores.items():
        print(f"-- {metric_name} --\n{score}\n")

def print_fold_scores(scores):
    for i, fold in enumerate(scores):
        print(f"### Fold {i + 1}:")
        print_dict_of_scores(fold)

def run_sehri_et_al_papers(model):
    from sklearn.metrics import accuracy_score, confusion_matrix
    from dataset.cwru.sehri_et_al import run_papers_experiment
    list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
    scores = run_papers_experiment(model,list_of_metrics=list_of_metrics)
    print_dict_of_scores(scores)

def run_sehri_et_al_papers_inspired_experiment(model):
    from sklearn.metrics import accuracy_score, confusion_matrix
    from dataset.cwru.sehri_et_al import run_papers_inspired_experiment
    list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
    scores = run_papers_inspired_experiment(model,list_of_metrics=list_of_metrics)
    print_fold_scores(scores)

def run_sehri_et_al_proposed_experiment(model):
    from sklearn.metrics import accuracy_score, confusion_matrix
    from dataset.cwru.sehri_et_al import run_proposed_experiment
    list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
    scores = run_proposed_experiment(model,list_of_metrics=list_of_metrics)
    print_fold_scores(scores)

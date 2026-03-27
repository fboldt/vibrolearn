from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
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

def run_sehri_et_al_experiment(model):
    from dataset.cwru.sehri_et_al import get_list_of_papers_X_y
    from sklearn.metrics import classification_report, confusion_matrix
    list_of_X_y = get_list_of_papers_X_y()
    for i, (X, y) in enumerate(list_of_X_y):
        print(f"Fold {i+1}:")
        model.fit(X, y)
        y_pred = model.predict(X)
        print(classification_report(y, y_pred))
        print(confusion_matrix(y, y_pred))
        print("-" * 40)
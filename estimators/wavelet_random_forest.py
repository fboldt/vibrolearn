from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from feature.extraction import WaveletPackage

def get_wavelet_random_forest(random_state = None):
    model = Pipeline(
        steps=[
            (
                "wavelet",
                WaveletPackage(
                    wavelet="db4",
                    mode="symmetric",
                    maxlevel=4,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return model

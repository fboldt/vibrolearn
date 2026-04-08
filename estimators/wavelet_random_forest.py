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

from estimators.base_classifier import BaseClassifier
class WaveletRandomForest(BaseClassifier):
    def __init__(self, random_state = None):
        self.model = get_wavelet_random_forest(random_state=random_state)
  
    def train(self, list_of_registers):
        X, y = self.load_function(list_of_registers)
        self.model.fit(X, y)
        return self
    
    def evaluate(self, list_of_registers, list_of_metrics):
        X, y = self.load_function(list_of_registers)
        y_pred = self.model.predict(X)
        scores = {}
        for metric in list_of_metrics:
            scores[metric.__name__] = metric(y, y_pred)
        return scores


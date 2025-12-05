from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from feature.wavelet_package import WaveletPackage

db30_smooth = {'wavelet': 'db30', 'mode': 'smooth', 'maxlevel': 9}
db36_constant = {'wavelet': 'db36', 'mode': 'constant', 'maxlevel': 6}
bior3_5 = {'wavelet': 'bior3.5', 'mode': 'periodization', 'maxlevel': 4}


class WaveletPackageFeatures(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.pipeline = Pipeline([
            ('feature_extractor', WaveletPackage(**bior3_5)),
            ('classifier', self.estimator)
        ])
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)

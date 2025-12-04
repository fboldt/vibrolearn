from sklearn.base import BaseEstimator, ClassifierMixin
from feature.AutoWaveletPackage import AutoWaveletPackage as FeatureExtractor
from sklearn.ensemble import RandomForestClassifier as Estimator


class AWPRF(BaseEstimator, ClassifierMixin):
    def __init__(self, trials=200):
        self.model = FeatureExtractor(estimator=Estimator(), trials=trials)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
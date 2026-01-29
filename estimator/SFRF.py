from sklearn.base import BaseEstimator, ClassifierMixin
from feature.StatisticalFeatures import StatisticalFeatures as FeatureExtractor
from sklearn.ensemble import RandomForestClassifier as Estimator


class SFRF(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = FeatureExtractor(estimator=Estimator())

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from feature.heterogeneous import Heterogeneous as FeatureExtractor
from feature.statistical_time import StatisticalTime
from feature.statistical_frequency import StatisticalFrequency
from feature.wavelet_package import WaveletPackage

featureExtractor = FeatureExtractor([
    StatisticalTime,
    StatisticalFrequency,
    WaveletPackage
])

class HeterogeneuousFeatures(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.pipeline = Pipeline([
            ('feature_extractor', featureExtractor),
            ('classifier', self.estimator)
        ])
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
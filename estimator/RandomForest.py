from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
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

class RandomForest(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.pipeline = Pipeline([
            ('feature_extractor', featureExtractor),
            ('classifier', RandomForestClassifier())
        ])
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
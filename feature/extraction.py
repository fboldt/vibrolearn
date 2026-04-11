from sklearn.base import BaseEstimator, ClassifierMixin
from feature.ConcatenateFeatures import ConcatenateFeatures
from feature.statistical_time import StatisticalTime
from feature.statistical_frequency import StatisticalFrequency
from feature.wavelet_package import WaveletPackage
from feature.flatten import Flatten

class FlattenFeatures(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.featureExtractor = Flatten()

    def fit(self, X, y):
        self.features = self.featureExtractor.fit(X, y)
        return self
    
    def transform(self, X):
        return self.features.transform(X)

class StatisticalFeatures(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.featureExtractor = ConcatenateFeatures([
            StatisticalTime, 
            StatisticalFrequency])

    def fit(self, X, y):
        self.features = self.featureExtractor.fit(X, y)
        return self
    
    def transform(self, X):
        return self.features.transform(X)


class HeterogeneousFeatures(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.featureExtractor = ConcatenateFeatures([
            StatisticalTime, 
            StatisticalFrequency, 
            WaveletPackage])

    def fit(self, X, y):
        self.features = self.featureExtractor.fit(X, y)
        return self
    
    def transform(self, X):
        return self.features.transform(X)
    

class WaveletFeatures(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.featureExtractor = WaveletPackage()

    def fit(self, X, y):
        self.features = self.featureExtractor.fit(X, y)
        return self
    
    def transform(self, X):
        return self.features.transform(X)
    
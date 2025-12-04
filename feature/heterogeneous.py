import numpy as np
from sklearn.base import TransformerMixin


class Heterogeneous(TransformerMixin):
    '''
    Extracts statistical features from both time and frequency domain also WaveletPackage.
    '''
    def __init__(self, extractors):
        self.extractors = extractors

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features_list = []
        for extractor in self.extractors:
            features = extractor().transform(X)
            features_list.append(features)
        features = np.concatenate(features_list, axis=1)
        return features
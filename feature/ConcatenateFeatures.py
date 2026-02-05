import numpy as np
from sklearn.base import TransformerMixin


class ConcatenateFeatures(TransformerMixin):
    '''
    Extracts diverse features from the input data and concatenates them into a single feature matrix.
    This class takes a list of feature extractor classes, fits each extractor to the data, and transforms the data using each extractor. The resulting feature matrices are then concatenated along the feature axis.
    '''
    def __init__(self, extractors):
        self.extractors = extractors

    def fit(self, X, y=None):
        self.fitted_extractors = []
        for extractor in self.extractors:
            fitted_extractor = extractor().fit(X, y)
            self.fitted_extractors.append(fitted_extractor)
        return self

    def transform(self, X, y=None):
        features_list = []
        for extractor in self.fitted_extractors:
            features = extractor.transform(X)
            features_list.append(features)
        features = np.concatenate(features_list, axis=1)
        return features

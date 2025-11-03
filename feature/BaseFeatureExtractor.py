import numpy as np
from sklearn.base import TransformerMixin

def transform_channels_to_features(X, extract_features):
    n_channels = X.shape[-1]
    features = []
    for i in range(n_channels):
        channel_features = extract_features(X[:, i])
        features.append(channel_features)
    channel_features = extract_features(X.reshape(X.shape[0], -1))
    features.append(channel_features)
    features = np.concatenate(features, axis=1)
    return features

class BaseFeatureExtractor(TransformerMixin):
  def __init__(self, extract_features):
    super().__init__()
    self.extract_features = extract_features
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    return transform_channels_to_features(X, extract_features=self.extract_features)
  





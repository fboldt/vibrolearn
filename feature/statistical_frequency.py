
import numpy as np
from sklearn.base import TransformerMixin
from feature.statistical_time import rms
from feature.utils import transform_channels_to_features


def extract_features(X):
    features = []
    for x in X:
      fx = np.absolute(np.fft.fft(x)) # transform x from time to frequency domain
      fc = np.mean(fx) # frequency center
      features.append([
                fc, # frequency center
                rms(fx), # RMS from the frequency domain
                rms(fx-fc), # Root Variance Frequency
                ])
    features = np.array(features)
    return features


class StatisticalFrequency(TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    return transform_channels_to_features(X, extract_features=extract_features)
  
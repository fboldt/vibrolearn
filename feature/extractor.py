import numpy as np
from sklearn.base import TransformerMixin
from feature.statistical_time import extract_features as stat_time_features
from feature.statistical_frequency import extract_features as stat_freq_features
from feature.wavelet_package import extract_features as wavelet_package_features

def transform_channels_to_features(X, extract_features):
    n_channels = X.shape[-1]
    features = []
    for i in range(n_channels):
        channel_features = extract_features(X[:, :, i])
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
  

class StatisticalTime(BaseFeatureExtractor):
  '''
  Extracts statistical features from the time domain.
  '''
  def __init__(self):
    super().__init__(extract_features=stat_time_features)


class StatisticalFrequency(BaseFeatureExtractor):
  '''
  Extracts statistical features from the frequency domain.
  '''
  def __init__(self):
    super().__init__(extract_features=stat_freq_features)


class StatisticalTimeFrequency(TransformerMixin):
  '''
  Extracts statistical features from both time and frequency domain.
  '''
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    st = StatisticalTime()
    stfeats = st.transform(X)
    sf = StatisticalFrequency()
    sffeats = sf.transform(X)
    features = np.concatenate((stfeats, sffeats), axis=1)
    return features


class WaveletPackage(BaseFeatureExtractor):
  '''
  Extracts Wavelet Package features.
  The features are calculated by the energy of the recomposed signal
  of the leaf nodes coefficients.
  '''
  def __init__(self):
    super().__init__(extract_features=wavelet_package_features)
  

class Heterogeneous(TransformerMixin):
  '''
  Extracts statistical features from both time and frequency domain also WaveletPackage.
  '''
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    st = StatisticalTime()
    stfeats = st.transform(X)
    sf = StatisticalFrequency()
    sffeats = sf.transform(X)
    wp = WaveletPackage()
    wpfeats = wp.transform(X)
    features = np.concatenate((stfeats, sffeats, wpfeats), axis=1)
    return features
  
  
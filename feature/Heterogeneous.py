
import numpy as np
from sklearn.base import TransformerMixin
from feature.StatisticalTime import StatisticalTime
from feature.StatisticalFrequency import StatisticalFrequency
from feature.WaveletPackage import WaveletPackage

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

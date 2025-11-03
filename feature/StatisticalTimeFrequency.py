
import numpy as np
from sklearn.base import TransformerMixin
from feature.StatisticalTime import StatisticalTime
from feature.StatisticalFrequency import StatisticalFrequency

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

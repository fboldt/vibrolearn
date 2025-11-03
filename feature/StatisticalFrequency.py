
import numpy as np
from feature.StatisticalTime import rms
from feature.BaseFeatureExtractor import BaseFeatureExtractor

def extract_features(X):
    features = []
    for x in X[:]:
      fx = np.absolute(np.fft.fft(x)) # transform x from time to frequency domain
      fc = np.mean(fx) # frequency center
      features.append([
                fc, # frequency center
                rms(fx), # RMS from the frequency domain
                rms(fx-fc), # Root Variance Frequency
                ])
    features = np.array(features)
    return features

class StatisticalFrequency(BaseFeatureExtractor):
  '''
  Extracts statistical features from the frequency domain.
  '''
  def __init__(self):
    super().__init__(extract_features=extract_features)


import numpy as np
from feature.statistical_time import rms


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

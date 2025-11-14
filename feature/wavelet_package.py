
import numpy as np
import pywt
from sklearn.base import TransformerMixin


def Energy(coeffs, k):
      return np.sqrt(np.sum(np.array(coeffs[-k]) ** 2)) / len(coeffs[-k])


def getEnergy(wp):
  coefs = np.asarray([n.data for n in wp.get_leaf_nodes(True)])
  return np.asarray([Energy(coefs,i) for i in range(2**wp.maxlevel)])


def extract_features(X, wavelet='db4', mode='symmetric', maxlevel=4):
    wp = [pywt.WaveletPacket(data=x, wavelet=wavelet, mode=mode, maxlevel=maxlevel) for x in X]
    features = np.array([getEnergy(wp_i) for wp_i in wp])
    return features

def wavelist(kind='discrete'):
    return pywt.wavelist(kind=kind)

class WaveletPackage(TransformerMixin):
    def __init__(self, wavelet='db4', mode='symmetric', maxlevel=4):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.maxlevel = maxlevel
    
    def transform_channels_to_features(self, X, extract_features):
        n_channels = X.shape[-1]
        features = []
        for i in range(n_channels):
            channel_features = extract_features(X[:, :, i])
            features.append(channel_features)
        features = np.concatenate(features, axis=1)
        return features

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        def wavelets(X):
            return extract_features(X, wavelet=self.wavelet, mode=self.mode, maxlevel=self.maxlevel)
        
        return self.transform_channels_to_features(X, extract_features=wavelets)
    
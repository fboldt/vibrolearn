
import numpy as np
import pywt

def Energy(coeffs, k):
      return np.sqrt(np.sum(np.array(coeffs[-k]) ** 2)) / len(coeffs[-k])

def getEnergy(wp):
  coefs = np.asarray([n.data for n in wp.get_leaf_nodes(True)])
  return np.asarray([Energy(coefs,i) for i in range(2**wp.maxlevel)])

def extract_features( X):
    features = np.array([getEnergy(pywt.WaveletPacket(data=x, wavelet='db4',
                                                mode='symmetric', maxlevel=4)
                                                ) for x in X])
    return features


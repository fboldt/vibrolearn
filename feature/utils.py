import numpy as np

def transform_channels_to_features(X, extract_features):
    n_channels = X.shape[-1]
    features = []
    for i in range(n_channels):
        channel_features = extract_features(X[:, :, i])
        features.append(channel_features)
    features = np.concatenate(features, axis=1)
    return features


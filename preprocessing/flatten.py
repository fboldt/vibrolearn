from sklearn.base import BaseEstimator, TransformerMixin

class Flatten(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)

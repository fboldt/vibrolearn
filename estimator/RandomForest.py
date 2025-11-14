from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from feature.extractor import Heterogeneous as FeatureExtractor
import optuna

class RandomForest(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.pipeline = Pipeline([
            ('feature_extractor', FeatureExtractor()),
            ('normalizer', Normalizer()),
            ('classifier', RandomForestClassifier())
        ])
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
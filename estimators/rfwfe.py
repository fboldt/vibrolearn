from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from feature.extractor import Heterogeneous as FeatureExtractor


model = Pipeline([
    ('feature_extractor', FeatureExtractor()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
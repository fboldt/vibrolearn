
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from feature.extractor import Heterogeneous as FeatureExtractor


model = Pipeline([
    ('feature_extractor', FeatureExtractor()),
    ('normalizer', Normalizer()),
    ('classifier', AdaBoostClassifier(
        estimator=RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    ))
])
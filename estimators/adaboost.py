
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from feature.StatisticalTime import StatisticalTime as FeatureExtractor


model = Pipeline([
    ('feature_extractor', FeatureExtractor()),
    ('classifier', AdaBoostClassifier(
        estimator=RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    ))
])
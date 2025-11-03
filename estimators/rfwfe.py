from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from feature.StatisticalTime import StatisticalTime as FeatureExtractor
# from feature.StatisticalFrequency import StatisticalFrequency as FeatureExtractor
# from feature.WaveletPackage import WaveletPackage as FeatureExtractor
# from feature.Heterogeneous import Heterogeneous as FeatureExtractor
# from feature.StatisticalTimeFrequency import StatisticalTimeFrequency as FeatureExtractor


model = Pipeline([
    # ('flatten', Flatten()),
    ('feature_extractor', FeatureExtractor()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
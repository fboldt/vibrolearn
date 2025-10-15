from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from feature.wavelet import WaveletPackage
from preprocessing.flatten import Flatten


model = Pipeline([
    ('flatten', Flatten()),
    ('feature_extractor', WaveletPackage()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
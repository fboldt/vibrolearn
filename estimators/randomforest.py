from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from preprocessing.flatten import Flatten 
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('flatten', Flatten()),
    ('normalizer', Normalizer()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

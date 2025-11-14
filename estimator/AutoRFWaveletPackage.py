from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from feature.wavelet_package import WaveletPackage, wavelist
from pprint import pprint
import optuna

def optimize_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, optimization_metric=optimize_metric, trials=20):
        self.model = None
        self.optimization_metric = optimization_metric
        self.trials = trials

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False)
        def objective(trial):
            wavelet = trial.suggest_categorical('wavelet', wavelist(kind='discrete'))
            mode = trial.suggest_categorical('mode', ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect', 'antisymmetric', 'antireflect'])
            maxlevel = trial.suggest_int('maxlevel', 2, 5)
            pipeline = Pipeline([
                ('feature_extractor', WaveletPackage(
                    wavelet=wavelet,
                    mode=mode,
                    maxlevel=maxlevel
                )),
                ('normalizer', Normalizer()),
                ('classifier', RandomForestClassifier())
            ])
            pipeline.fit(X_train, y_train)
            ypred = pipeline.predict(X_val)
            performance_metric = self.optimization_metric(y_val, ypred)
            return performance_metric
        # optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.trials)
        pprint(study.best_params)
        feature_extractor = WaveletPackage(**study.best_params)
        self.pipeline = Pipeline([
            ('feature_extractor', feature_extractor),
            ('normalizer', Normalizer()),
            ('classifier', RandomForestClassifier())
        ])
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
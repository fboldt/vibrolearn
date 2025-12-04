from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from feature.wavelet_package import WaveletPackage, wavelist
from pprint import pprint
import optuna


class AutoWaveletPackage(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, trials=200, verbose=True):
        self.estimator = estimator
        self.trials = trials
        self.verbose = verbose

    def fit(self, X, y):
        def objective(trial):
            wavelet = trial.suggest_categorical('wavelet', 
                                                wavelist(kind='discrete'))
            mode = trial.suggest_categorical('mode', 
                                             ['zero', 
                                              'constant', 
                                              'symmetric', 
                                              'periodic', 
                                              'smooth', 
                                              'periodization', 
                                              'reflect', 
                                              'antisymmetric', 
                                              'antireflect'])
            maxlevel = trial.suggest_int('maxlevel', 1, 10)
            pipeline = Pipeline([
                ('feature_extractor', WaveletPackage(
                    wavelet=wavelet,
                    mode=mode,
                    maxlevel=maxlevel
                )),
                ('classifier', self.estimator)
            ])
            scores = cross_validate(pipeline, X, y, cv=3, scoring='f1_macro')
            score = scores['test_score'].mean()
            return score
        
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.trials)
        pprint(study.best_params)
        feature_extractor = WaveletPackage(**study.best_params)
        self.pipeline = Pipeline([
            ('feature_extractor', feature_extractor),
            ('classifier', self.estimator)
        ])
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
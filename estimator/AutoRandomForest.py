from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from feature.extractor import Heterogeneous as FeatureExtractor
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
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf)
            pipeline = Pipeline([
                ('feature_extractor', FeatureExtractor()),
                ('normalizer', Normalizer()),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            ypred = pipeline.predict(X_val)
            performance_metric = self.optimization_metric(y_val, ypred)
            return performance_metric
        # optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.trials)
        model = RandomForestClassifier(**study.best_params)
        self.pipeline = Pipeline([
            ('feature_extractor', FeatureExtractor()),
            ('normalizer', Normalizer()),
            ('classifier', model)
        ])
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
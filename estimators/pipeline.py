from sklearn.pipeline import Pipeline as SklearnPipeline
from dataset.loader import vanilla
from timed_decorator.simple_timed import timed


@timed(return_time=True, use_seconds=True)
def timed_load(loader, registers, experimental_setup):
    return loader(registers, experimental_setup)


@timed(return_time=True, use_seconds=True)
def timed_fit(pipe, X, y, domains=None):
    if domains is not None:
        return pipe.fit(X, y, feature_selection__domains=domains)
    return pipe.fit(X, y)


@timed(return_time=True, use_seconds=True)
def timed_predict(pipe, X):
    return pipe.predict(X)


class Pipeline():
    def __init__(self, steps):
        self.pipe = SklearnPipeline(steps)
        self.scores = {}
        self.train_loader = vanilla
        self.evaluate_loader = vanilla
  
    def train(self, list_of_registers, experimental_setup):
        self.experimental_setup = experimental_setup
        loaded_data, load_time = timed_load(
            self.train_loader,
            list_of_registers,
            self.experimental_setup
        )
        if len(loaded_data) == 3:
            X, y, domains = loaded_data
        else:
            X, y = loaded_data
            domains = None
        self.scores["load_data_time"] = load_time
        _, training_time = timed_fit(self.pipe, X, y, domains=domains)
        self.scores["training_time"] = training_time
        return self
    
    def evaluate(self, list_of_registers, list_of_metrics):
        # (X, y, _), _ = timed_load(self.evaluate_loader, list_of_registers, self.experimental_setup)
        loaded_data, _ = timed_load(
            self.evaluate_loader,
            list_of_registers,
            self.experimental_setup
        )
        if len(loaded_data) == 3:
            X, y, _ = loaded_data
        else:
            X, y = loaded_data
        y_pred, prediction_time = timed_predict(self.pipe, X)
        self.scores["prediction_time"] = prediction_time
        scores = {}
        for metric in list_of_metrics:
            scores[metric.__name__] = metric(y, y_pred)
        self.scores = self.scores | scores
        return self.scores

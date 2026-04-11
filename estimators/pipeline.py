from sklearn.pipeline import Pipeline as SklearnPipeline
import numpy as np

class Pipeline():
    def __init__(self, steps):
        self.pipe = SklearnPipeline(steps)
        self.load_function = None

    def set_load_function(self, load_function):
        self.load_function = load_function
  
    def train(self, list_of_registers):
        X, y = self.load_function(list_of_registers) if self.load_function else (np.array([[0, 0]]), None)
        self.pipe.fit(X, y)
        return self
    
    def evaluate(self, list_of_registers, list_of_metrics):
        X, y = self.load_function(list_of_registers) if self.load_function else (np.array([[0, 0]]), None)
        y_pred = self.pipe.predict(X)
        scores = {}
        for metric in list_of_metrics:
            scores[metric.__name__] = metric(y, y_pred)
        return scores
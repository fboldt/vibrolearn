from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    def set_load_function(self, load_function):
        self.load_function = load_function

    @abstractmethod
    def train(self, list_of_registers):
        return self

    @abstractmethod
    def evaluate(self, list_of_registers, list_of_metrics):
        scores = {}
        return scores
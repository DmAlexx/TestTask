# models/base.py

from abc import ABC, abstractmethod

# Interface for MNIST classification models.
class MnistClassifierInterface(ABC):

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
       # Trains the model.
        pass

    @abstractmethod
    def predict(self, X_test):
        # Predicts class labels for input data.
        pass

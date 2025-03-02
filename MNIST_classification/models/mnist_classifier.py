# models/mnist_classifier.py

from models.random_forest import RandomForestMnist
from models.neural_network import NeuralNetworkMnist
from models.cnn import CnnMnist

class MnistClassifier:
    """
    A wrapper class to select and use different MNIST classification models.
    """

    def __init__(self, algorithm):
        """
        Initializes the classifier with the selected algorithm.
        
        :param algorithm: The algorithm to use. Options: "rf" (Random Forest), "nn" (Fully Connected NN), "cnn" (Convolutional NN).
        """
        if algorithm == "rf":
            self.model = RandomForestMnist()
        elif algorithm == "nn":
            self.model = NeuralNetworkMnist()
        elif algorithm == "cnn":
            self.model = CnnMnist()
        else:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the selected model.
        
        :param X: Training features (images).
        :param y: Training labels.
        :param kwargs: Additional arguments (e.g., epochs for NN/CNN).
        """
        self.model.train(X_train, y_train, X_val, y_val)

    def predict(self, X_test):
        """
        Makes predictions using the selected model.
        
        :param X: Input images for classification.
        :return: Predicted class labels.
        """
        return self.model.predict(X_test)

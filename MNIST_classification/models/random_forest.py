# models/random_forest.py

from models.base import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier

# Class for MNIST image classification using Random Forest.
class RandomForestMnist(MnistClassifierInterface):

    def __init__(self):
        # Initializes the Random Forest model.
        self.model = RandomForestClassifier(n_estimators=25, random_state=42)

    def train(self, X_train, y_train, X_val, y_val):

        # Fit the model on training data
        self.model.fit(X_train, y_train)

        # Evaluate on validation set
        val_accuracy = self.model.score(X_val, y_val)
        print(f"Validation Accuracy (RF): {val_accuracy:.4f}")

    def predict(self, X_test):
        # Makes predictions on the test data.
        return self.model.predict(X_test)
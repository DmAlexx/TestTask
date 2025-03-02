# models/neural_network.py

from models.base import MnistClassifierInterface
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

class NeuralNetworkMnist(MnistClassifierInterface):
    """
    Fully connected neural network for MNIST classification.
    """

    def __init__(self):
        # Initialize the fully connected neural network model.

        input_shape=(28, 28)
        num_classes=10
        hidden_units=128

        self.model = Sequential([
            Flatten(input_shape=input_shape),
            Dense(256, activation="relu"),
            Dropout(0.3),  
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax")
        ])


        # Compile the model
        self.model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def train(self, X_train, y_train, X_val, y_val):
        # Train the neural network model.

        epochs=11
        batch_size=32

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val))

        def plot_history(history, epochs=epochs):
            h = history.history
            epochs = range(len(h['loss']))

            plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
            plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
            plt.legend(['Train', 'Validation'])
            plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-', epochs, h['val_accuracy'], '.-')
            plt.grid(True), plt.xlabel('epochs'), plt.ylabel('Accuracy')
            plt.legend(['Train', 'Validation'])
        
            print('Train Acc     ', h['accuracy'][-1])
            print('Validation Acc', h['val_accuracy'][-1])
    
        plot_history(history)

    def predict(self, X_test):
        # Predictions on the test data.
 
        predictions = self.model.predict(X_test)
        return predictions.argmax(axis=1)  # Get the class with highest probability

# Forked from:      Patrick Loeber
#                   https://www.youtube.com/playlist?list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E
#                   https://github.com/patrickloeber/MLfromscratch/blob/7f0f18ada1f75d1999a5206b5126459d51f73dce/mlfromscratch/logistic_regression.py

# Updated to handle multiple classes and be compatible with sklearn analysis

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Derive from sklearn class for compatibility with sklearn cross validation
class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.001, n_iters=1000, weights=None, biases=None, classes=None):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = weights
        self.biases = biases
        self.classes = classes

    def gradient_descent(self, X, y, weight, bias, learning_rate, num_iterations):
        n_samples, n_features = X.shape
        # gradient descent
        for i in range(num_iterations):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, weight) + bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            weight -= learning_rate * dw
            bias -= learning_rate * db
        return weight, bias
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        # Initialize parameters for each class
        self.weights = np.zeros((num_classes, n_features))
        self.biases = np.zeros(num_classes)

        # init parameters
        weight = np.zeros(n_features)
        bias = 0

        # Train a binary classifier for each class (OvR)
        for idx, c in enumerate(self.classes):
            y_binary = (y == c).astype(int)
            self.weights[idx], self.biases[idx] = self.gradient_descent(X, y_binary, weight, bias, self.lr, self.n_iters)


    def predict(self, X):
        linear_model = np.dot(X, self.weights.T) + self.biases
        y_predicted = self._sigmoid(linear_model)
        predictions = np.argmax(y_predicted, axis=1)
        return self.classes[predictions]
    
    # Imitate sklearn's score() function
    def score(self, X_test, y_true):
        y_pred = self.predict(X_test)
        return np.sum(y_true == y_pred) / len(y_true)

    # Imitate sklearn's get_params() to work with xvalidation
    def get_params(self, deep=True):
        return {"learning_rate": self.lr,
                "n_iters": self.n_iters,
                "weights": self.weights,
                "biases": self.biases,
                "classes": self.classes
                }

    # Imitate sklearn's set_params() to work with xvalidation
    def set_params(self, **params):
        # Set the parameters of the classifier
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print(f"LR classification accuracy: {accuracy(y_test, predictions):.3f}")
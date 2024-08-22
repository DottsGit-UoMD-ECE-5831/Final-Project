import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.01, n_iters=2000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.biases = None
        self.classes = None
        self.train_accuracy = []

    def gradient_descent(self, X, y, weight, bias, learning_rate, num_iterations):
        n_samples, n_features = X.shape
        for i in range(num_iterations):
            linear_model = np.dot(X, weight) + bias
            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            weight -= learning_rate * dw
            bias -= learning_rate * db

            # Calculate and store accuracy at this iteration
            predictions = np.round(y_predicted)
            accuracy = np.sum(y == predictions) / len(y)
            self.train_accuracy.append(accuracy)
            
        return weight, bias
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        self.weights = np.random.randn(num_classes, n_features) * 0.01  # Random initialization
        self.biases = np.zeros(num_classes)
        self.train_accuracy = []

        for idx, c in enumerate(self.classes):
            y_binary = (y == c).astype(int)
            self.weights[idx], self.biases[idx] = self.gradient_descent(X, y_binary, self.weights[idx], self.biases[idx], self.lr, self.n_iters)

    def predict(self, X):
        linear_model = np.dot(X, self.weights.T) + self.biases
        y_predicted = self._sigmoid(linear_model)
        predictions = np.argmax(y_predicted, axis=1)
        return self.classes[predictions]

    def score(self, X_test, y_true):
        y_pred = self.predict(X_test)
        return accuracy_score(y_true, y_pred)

    def get_params(self, deep=True):
        return {"lr": self.lr, "n_iters": self.n_iters}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example implementation and testing
if __name__ == "__main__":
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regressor = LogisticRegression(lr=0.01, n_iters=2000)  # Adjusted learning rate and iterations
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print(f"LR classification accuracy: {accuracy:.3f}")
    print(f"LR classification precision: {precision:.3f}")
    print(f"LR classification recall: {recall:.3f}")

    # Cross-validation
    cv_scores = cross_val_score(regressor, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average cross-validation score: {np.mean(cv_scores):.3f}")

    # Plot the learning curve for accuracy
    plt.plot(range(1, len(regressor.train_accuracy) + 1), regressor.train_accuracy, label='Training Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve for Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

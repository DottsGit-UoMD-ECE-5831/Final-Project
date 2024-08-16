# Matthew Kwan
# 8/15/24

# Portions derived from:     Patrick Loeber
#                            https://www.youtube.com/playlist?list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E
#                            https://github.com/patrickloeber/MLfromscratch/blob/7f0f18ada1f75d1999a5206b5126459d51f73dce/mlfromscratch/kmeans.py

from logistic_regression import LogisticRegression
from preprocessing import Preprocessor
import numpy as np
    
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    preprocessor = Preprocessor()
    X, y, iris_df = preprocessor.preprocess_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print("LR classification accuracy:", accuracy(y_test, predictions))
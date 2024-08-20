# Matthew Kwan
# 8/15/24

# Portions derived from:     Patrick Loeber
#                            https://www.youtube.com/playlist?list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E
#                            https://github.com/patrickloeber/MLfromscratch/blob/7f0f18ada1f75d1999a5206b5126459d51f73dce/mlfromscratch/kmeans.py

# Adding directories for imports
import sys
import os
from evaluation import Evaluation
sys.path.append(os.path.join(os.path.dirname(__file__), 'Part_1_Preprocessing'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Part_3_Logistical_Regression'))

from Part_1_Preprocessing.preprocessing import Preprocessor
from Part_3_Logistical_Regression.logistic_regression import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split

def accuracy(y_true, y_pred) -> float:
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

if __name__ == "__main__":
    preprocessor = Preprocessor()
    X, y, iris_df = preprocessor.preprocess_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Evaluate the model's performance
    e = Evaluation()
    e.evaluate(y_test, y_pred)
    e.kfold_cross_validate(X, y, classifier=regressor)
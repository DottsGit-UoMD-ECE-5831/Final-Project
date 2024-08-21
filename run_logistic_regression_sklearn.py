import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from evaluation import Evaluation
sys.path.append(os.path.join(os.path.dirname(__file__), 'Part_1_Preprocessing'))
from Part_1_Preprocessing.preprocessing import Preprocessor


if __name__ == "__main__":
    preprocessor = Preprocessor()
    X, y, iris_df = preprocessor.preprocess_iris()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    
    # Initialize the Logistic Regression model
    # Use One-vs-Rest (OvR) to match from-scratch implementation
    clf = LogisticRegression(multi_class='ovr')
    
    # Fit the model to the training data
    clf.fit(X_train, y_train)
    
    # Predict the labels for the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the model's performance
    e = Evaluation()
    e.evaluate(y_test, y_pred)
    e.kfold_cross_validate(X, y, classifier=clf)
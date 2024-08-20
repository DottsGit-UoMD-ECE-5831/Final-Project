from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Evaluation:
    def __init__(self):
        pass

    # y_test: output format of sklearn's train_test_split()
    # y_pred: output format of sklearn's LogisticRegression predict()
    def evaluate(self, y_test, y_pred):
        # Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, zero_division=0)
        
        # Print the results (incl accuracy, precision, and recall)
        print(f"Test Set Accuracy: {accuracy:.3f}")
        print("\nConfusion Matrix:\n", conf_matrix)
        print("\nClassification Report:\n", class_report)
        
    # classifier contains fit() and predict() and score()
    def kfold_cross_validate(self, X, y, classifier, num_folds=12, random_state=321):
        # Cross-validation
        # Code snippet derived from: 
        # https://www.geeksforgeeks.org/cross-validation-machine-learning/
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        cross_val_results = cross_val_score(classifier, X, y, cv=kf)
        # Format each element to 3 decimal places
        xval_mean = cross_val_results.mean()
        cross_val_results_rounded = [f"{x:.3f}" for x in cross_val_results]
        print(f'KFold Cross-Validation Results (Accuracy): {cross_val_results_rounded}')
        print(f'Mean Accuracy: {xval_mean:.3f}')
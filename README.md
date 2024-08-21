## Usage
This project looks at classifying the iris dataset with the features 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', and 'petal width (cm)'

Run the python files prepended with run_, e.g.,
```
python run_kmeans.py
python run_logistic_regression_scratch.py
python run_logistic_regression_sklearn.py
```
To adjust parameters such as plotting, adjust the inputs at the top of the source code.

## Credit
Much of this work is derived from other sources, see files for sources.

## Example Output
Histogram output of the preprocessing:
![histograms](images/histograms.png "Histogram of the data")

Scatter plot output of the preprocessing:
![scatterplots](images/scatter-plots.png "Scatter plots of the data")

Example output of KMeans between `sepal width (cm)` and `petal width (cm)`:
![kmeans](images/kmeans.png "KMeans scatter plot - sepal width vs petal width")

Example output of Logistic Regression from scratch using One-vs-Rest (OvR):
```
Test Set Accuracy: 0.956

Confusion Matrix:
 [[16  0  0]
 [ 0 16  1]
 [ 0  1 11]]

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       0.94      0.94      0.94        17
           2       0.92      0.92      0.92        12

    accuracy                           0.96        45
   macro avg       0.95      0.95      0.95        45
weighted avg       0.96      0.96      0.96        45

KFold Cross-Validation Results (Accuracy): ['1.000', '1.000', '0.923', '1.000', '0.846', '0.923', '0.917', '0.833', '1.000', '1.000', '0.917', '0.917']
Mean Accuracy: 0.940
```

Example output of Logistic Regression from sklearn using One-vs-Rest (OvR):
```
Test Set Accuracy: 0.956

Confusion Matrix:
 [[18  0  0]
 [ 0 10  0]
 [ 0  2 15]]

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        18
           1       0.83      1.00      0.91        10
           2       1.00      0.88      0.94        17

    accuracy                           0.96        45
   macro avg       0.94      0.96      0.95        45
weighted avg       0.96      0.96      0.96        45

KFold Cross-Validation Results (Accuracy): ['0.846', '0.846', '0.846', '0.923', '0.846', '0.923', '0.833', '0.750', '1.000', '0.833', '0.917', '1.000']
Mean Accuracy: 0.880
```
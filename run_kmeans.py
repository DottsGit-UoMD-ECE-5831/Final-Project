# Matthew Kwan
# 8/15/24

# Portions derived from:     Patrick Loeber
#                            https://www.youtube.com/playlist?list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E
#                            https://github.com/patrickloeber/MLfromscratch/blob/7f0f18ada1f75d1999a5206b5126459d51f73dce/mlfromscratch/kmeans.py

from kmeans import KMeans
from preprocessing import Preprocessor
import numpy as np
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # INPUT: Select 2 feature indices to plot on the 2D plot: 0, 1, 2, or 3
    # (Iris dataset contains 4 features)
    PLOT_FEATURE_1 = 1
    PLOT_FEATURE_2 = 3
    # /INPUT
    
    preprocessor = Preprocessor()
    X, y, iris_df = preprocessor.preprocess_iris()

    print(X.shape)

    num_clusters = len(np.unique(y))
    print(num_clusters)

    k = KMeans(K=num_clusters, max_iters=150, plot_steps=True, feature_index_1=PLOT_FEATURE_1, feature_index_2=PLOT_FEATURE_2)
    y_pred = k.predict(X)
    
    # TODO: add the x and y labels to plots based on iris_df.columns
    k.plot()
    
    # Accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
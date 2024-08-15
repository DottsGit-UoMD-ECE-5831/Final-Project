# Derived from:     Alireza Mohammadi
#                   amohmmad@umich.edu
#                   ECE5831 Final Project Help File

"""
This script visualizes the Iris dataset using histograms and pairwise scatter plots.

The Iris dataset is a classic dataset in machine learning and statistics, which consists of 150 samples from each of three species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the lengths and widths of the sepals and petals.

The script performs the following steps:
1. Load the Iris dataset and convert it to a Pandas DataFrame.
2. Add a categorical column for the species names.
3. Set the aesthetic style of the plots using Seaborn.
4. Create histograms for each feature with KDE (Kernel Density Estimate).
5. Create pairwise scatter plots to show the relationships between features, colored by species.
"""
# https://www.researchgate.net/publication/342859543_A_Comparative_Study_of_Linear_Regression_and_Regression_Tree/figures?lo=1

# Import necessary libraries
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For advanced plotting and visualization
from sklearn.datasets import load_iris  # To load the Iris dataset
import pandas as pd  # For data manipulation and analysis
from sklearn.preprocessing import LabelEncoder # To convert from species to ints

class Preprocessor:
    def __init__(self, plot=False):
        self.plot = plot

    def preprocess_iris(self):
        # Load the Iris dataset
        iris = load_iris()  # Load the dataset from sklearn
        # Convert the dataset into a Pandas DataFrame with appropriate column names
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        # Add a new column for the species, converting numeric target to categorical names
        iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        if (self.plot):
            # Set the aesthetic style of the plots
            sns.set(style="whitegrid")  # Use a white grid background for the plots
            
            # Create histograms for each feature
            plt.figure(figsize=(12, 8))  # Create a figure with a specified size
            # Iterate through each feature to create a subplot for its histogram
            for index, feature_name in enumerate(iris.feature_names):
                plt.subplot(2, 2, index + 1)  # Create a 2x2 grid of subplots
                sns.histplot(iris_df[feature_name], kde=True, bins=20)  # Create a histogram with KDE and 20 bins
                plt.title('Histogram of ' + feature_name)  # Set the title of the subplot
                plt.xlabel(feature_name)  # Label the x-axis with the feature name
                plt.ylabel('Count')  # Label the y-axis as 'Count'
            plt.tight_layout()  # Adjust subplots to fit into the figure area
            plt.show()  # Display the figure
            
            # Create scatter plots for each pair of features
            # Use pairplot to create a grid of scatter plots, colored by species
            sns.pairplot(iris_df, hue="species", markers=["o", "s", "D"])
            # Add a super title to the pairplot figure
            plt.suptitle('Pairwise scatter plots of the Iris features', verticalalignment='bottom')
            plt.show()  # Display the pairplot figure
            
        # Convert target names to integers
        y_names = iris_df['species']  # Target column
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_names)
        # Convert from dataframe to X and y
        X = iris_df.drop('species', axis=1).values  # Features (everything except the target column)
        return X, y, iris_df
            
if __name__ == "__main__":
    p = Preprocessor(plot=False)
    X, y, iris_df = p.preprocess_iris()
#Part 4 - Neutron Network: [4]input->[100]HiddenLayer*8->[3]output----------------------

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
import seaborn as sns
import numpy as np

# Load and preprocess the Iris dataset
def load_and_preprocess_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Define and train the neural network
def train_nn(X_train, y_train):
    print("Training the neural network...")
    nn = MLPClassifier(hidden_layer_sizes=(100,)*8, activation='relu', max_iter=200, random_state=42, verbose=True)
    nn.fit(X_train, y_train)
    return nn

# Plot neural network architecture
def plot_neural_network(layers):
    fig, ax = plt.subplots(figsize=(12, 8))
    layer_width = 0.1
    layer_gap = 0.2
    neuron_radius = 0.05

    def plot_layer(layer_x, layer_y, color):
        for y in layer_y:
            ax.add_patch(patches.Circle((layer_x, y), neuron_radius, color=color, zorder=5))

    input_layer_x = 0
    input_layer_y = [i / (max(layers[0], 1)) for i in range(layers[0])]
    plot_layer(input_layer_x, input_layer_y, 'blue')

    for i, layer_size in enumerate(layers[1:-1]):
        layer_x = (i + 1) * (layer_width + layer_gap)
        layer_y = [j / (max(layer_size, 1)) for j in range(layer_size)]
        plot_layer(layer_x, layer_y, 'green')

        for prev_y in input_layer_y:
            for curr_y in layer_y:
                ax.plot([input_layer_x, layer_x], [prev_y, curr_y], color='grey', alpha=0.5)

        input_layer_x = layer_x
        input_layer_y = layer_y

    output_layer_x = (len(layers) - 1) * (layer_width + layer_gap)
    output_layer_y = [i / (max(layers[-1], 1)) for i in range(layers[-1])]
    plot_layer(output_layer_x, output_layer_y, 'red')

    for prev_y in input_layer_y:
        for curr_y in output_layer_y:
            ax.plot([input_layer_x, output_layer_x], [prev_y, curr_y], color='grey', alpha=0.5)

    ax.set_xlim(-0.1, (len(layers) - 1) * (layer_width + layer_gap) + 0.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title('Neural Network Architecture with 8 Hidden Layers')
    plt.show()

# Plot learning curves
def plot_learning_curves(nn, X_train, y_train):
    print("Plotting learning curves...")
    train_sizes, train_scores, validation_scores = learning_curve(nn, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

    plt.figure(figsize=(12, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score', color='blue')
    plt.plot(train_sizes, np.mean(validation_scores, axis=1), label='Cross-validation score', color='green')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves (Neural Network)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot confusion matrix
def plot_confusion_matrices(nn, X_train, y_train, X_test, y_test):
    print("Plotting confusion matrices...")
    y_train_pred = nn.predict(X_train)
    y_test_pred = nn.predict(X_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax[0], xticklabels=iris.target_names, yticklabels=iris.target_names)
    ax[0].set_title('Training Confusion Matrix')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')

    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax[1], xticklabels=iris.target_names, yticklabels=iris.target_names)
    ax[1].set_title('Testing Confusion Matrix')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('True')

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        nn = train_nn(X_train, y_train)
        
        # Plot architecture
        plot_neural_network([4] + [100] * 8 + [3])
        
        # Plot learning curves
        plot_learning_curves(nn, X_train, y_train)
        
        # Plot confusion matrices
        plot_confusion_matrices(nn, X_train, y_train, X_test, y_test)
        
    except Exception as e:
        print(f"An error occurred: {e}")

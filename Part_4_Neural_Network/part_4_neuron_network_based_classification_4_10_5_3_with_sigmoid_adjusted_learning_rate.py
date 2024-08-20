#Part 4 Neuron Network - based Classification 4+10+5+3 with Sigmoid adjusted learning rate

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load and split the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network architecture with sigmoid activation function and improved parameters
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 5),
    activation='logistic',  # Sigmoid activation function
    max_iter=1,
    warm_start=True,
    random_state=42,
    learning_rate_init=0.01,  # Adjust learning rate
    alpha=0.0001  # Regularization parameter
)

# Initialize lists to collect loss values and accuracy scores
train_loss = []
train_accuracy = []

# Train the model and collect learning curves
max_iterations = 1000
for i in range(max_iterations):
    mlp.fit(X_train, y_train)
    train_loss.append(mlp.loss_)

    # Compute training accuracy
    train_accuracy.append(mlp.score(X_train, y_train))

# After training, compute cross-validation scores
cv_scores = cross_val_score(
    MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic',
                   max_iter=max_iterations, random_state=42,
                   learning_rate_init=0.01, alpha=0.0001),
    X, y, cv=5, scoring='accuracy'
)

# Plot learning curves
plt.figure(figsize=(12, 6))
plt.plot(range(1, max_iterations + 1), train_loss, label='Training Loss', color='r')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Function Convergence')
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy scores
plt.figure(figsize=(12, 6))
plt.plot(range(1, max_iterations + 1), train_accuracy, label='Training Accuracy', color='b')
plt.axhline(y=np.mean(cv_scores), color='g', linestyle='--', label='Cross-Validation Accuracy (mean)')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy Scores During Training')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model
y_pred_train = mlp.predict(X_train)
y_pred_test = mlp.predict(X_test)

print("Training Classification Report:")
print(classification_report(y_train, y_pred_train))

print("Testing Classification Report:")
print(classification_report(y_test, y_pred_test))

# Compute confusion matrices
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

# Plot confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax[0], xticklabels=iris.target_names, yticklabels=iris.target_names)
ax[0].set_title('Training Confusion Matrix')
ax[0].set_xlabel('Predicted Label')
ax[0].set_ylabel('True Label')

sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax[1], xticklabels=iris.target_names, yticklabels=iris.target_names)
ax[1].set_title('Testing Confusion Matrix')
ax[1].set_xlabel('Predicted Label')
ax[1].set_ylabel('True Label')

plt.tight_layout()
plt.show()

# Display the neural network architecture
def plot_nn_architecture(layers):
    fig, ax = plt.subplots(figsize=(16, 10))  # Increased figure size for more room
    layer_width = 1.0
    layer_height = 1.0
    padding = 0.5  # Padding between layers

    # Compute total height for axis limits
    total_height = 1.0
    for size in layers:
        total_height = max(total_height, size * layer_height / size + 0.5)

    ax.set_xlim(-0.5, len(layers) * (layer_width + padding) - 0.5)
    ax.set_ylim(-0.1, total_height)  # Ensure enough room for all layers

    prev_layer_size = layers[0]
    y_prev = 0.5 * (prev_layer_size - 1) / prev_layer_size

    for i, layer_size in enumerate(layers):
        x = i * (layer_width + padding)
        y = 0.5 * (layer_size - 1) / layer_size
        for j in range(layer_size):
            circle = plt.Circle((x, y + j * layer_height / layer_size), 0.1, color='skyblue', ec='black', lw=0.5)
            ax.add_artist(circle)
            if i > 0:
                for k in range(prev_layer_size):
                    line = plt.Line2D([x - (layer_width + padding), x], [y_prev + k * layer_height / prev_layer_size, y + j * layer_height / layer_size], color='gray', lw=0.5)
                    ax.add_artist(line)
        prev_layer_size = layer_size
        y_prev = y

    # Add layer labels with extra padding to avoid clipping
    for i, size in enumerate(layers):
        ax.text(i * (layer_width + padding), -0.2, f'Layer {i+1}\n({size} neurons)', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    ax.set_aspect('equal')
    ax.axis('off')

    plt.title('Neural Network Architecture')
    plt.show()

plot_nn_architecture([4, 10, 5, 3])  # Input layer (4), hidden layers (10, 5), output layer (3)

# Print loss function norm
print(f'Norm of the Loss Function: {np.linalg.norm(train_loss):.4f}')

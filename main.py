import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.ndimage import rotate, shift, zoom


# Configuration dictionary for flexible customization
config = {
    'input_size': 784,
    'hidden_layers': [128, 64, 32],
    'output_size': 10,
    'activation_functions': ['sigmoid', 'relu', 'relu'],
    'learning_rate': 0.001,
    'num_epochs': 15,
    'batch_size': 64,
    'optimizer': 'adam',  # Choose between 'adam' and 'rmsprop'
    'beta1': 0.9,  # For Adam
    'beta2': 0.999,  # For Adam
    'epsilon': 1e-8,  # Small value to prevent division by zero
    'decay_rate': 0.9,  # For RMSprop
    'test_size': 0.2,  # Test set size
    'val_size': 0.25,  # Validation size (from remaining train data)
    'random_state': 42,  # Random state for reproducibility
    'patience': 5,  # Patience for early stopping
    'use_data_augmentation': True,  # Activate or deactivate data augmentation
    'use_dropout': True,  # Activate or deactivate dropout
    'dropout_rate': 0.1  # Dropout probability
}

# Load MNIST data and convert to numpy arrays
mnist = fetch_openml('mnist_784')
X = mnist.data.to_numpy() / 255.0  # Normalize inputs and convert to numpy
y = mnist.target.astype(int).to_numpy()  # Convert to numpy array

# One-hot encoding for labels
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

# Train-test split (initial split into train + validation and test sets)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_one_hot, 
    test_size=config['test_size'], 
    random_state=config['random_state'], 
    stratify=y
)

# Further split train + validation set into actual train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, 
    test_size=config['val_size'], 
    random_state=config['random_state'], 
    stratify=np.argmax(y_temp, axis=1)
)

def augment_image(image):
    """
    Performs simple data augmentation on an image.
    Ensures the output is a 28x28 flattened image.
    """
    image = image.reshape(28, 28)  # Reshape for MNIST
    # Random rotation
    if np.random.rand() > 0.5:
        image = rotate(image, angle=np.random.uniform(-15, 15), reshape=False)
    # Random shift
    if np.random.rand() > 0.5:
        image = shift(image, shift=np.random.uniform(-2, 2, size=2))
    # Random zoom
    if np.random.rand() > 0.5:
        zoom_factor = np.random.uniform(0.9, 1.1)
        image = zoom(image, zoom_factor)
        # Crop or pad to ensure shape is 28x28
        if zoom_factor < 1:
            pad_width = ((0, 28 - image.shape[0]), (0, 28 - image.shape[1]))
            image = np.pad(image, pad_width, mode='constant', constant_values=0)
        elif zoom_factor > 1:
            crop_size = min(image.shape[0], 28)
            start = (image.shape[0] - crop_size) // 2
            image = image[start:start+28, start:start+28]
    
    # Clip pixel values and flatten
    image = np.clip(image, 0, 1).reshape(-1)
    return image


# Apply data augmentation only if activated
if config['use_data_augmentation']:
    X_train = np.array([augment_image(img) for img in X_train])

# Dropout function
def apply_dropout(A, dropout_rate):
    """
    Applies dropout to the activations.
    """
    if config['use_dropout']:
        mask = np.random.rand(*A.shape) > dropout_rate
        A *= mask  # Set some activations to 0
        A /= (1 - dropout_rate)  # Scale remaining activations
    return A

# Activation functions and their derivatives
def relu(x):
    """
    Computes the ReLU (Rectified Linear Unit) activation function.
    ReLU returns the input value if positive; otherwise, returns zero.
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Computes the derivative of the ReLU activation function.
    Returns 1 for positive inputs and 0 for non-positive inputs.
    """
    return (x > 0).astype(float)

def sigmoid(x):
    """
    Computes the sigmoid activation function.
    Maps input values to a range between 0 and 1.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Computes the derivative of the sigmoid activation function.
    Helps in backpropagation by measuring the slope of the sigmoid function.
    """
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x):
    """
    Computes the hyperbolic tangent activation function.
    Maps input values to a range between -1 and 1.
    """
    return np.tanh(x)

def tanh_derivative(x):
    """
    Computes the derivative of the hyperbolic tangent activation function.
    Returns the gradient of tanh for backpropagation.
    """
    return 1 - np.tanh(x)**2

# Dictionary of activation functions
activation_functions = {
    'relu': (relu, relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative)
}

# Weight initialization
np.random.seed(config['random_state'])
layers = [config['input_size']] + config['hidden_layers'] + [config['output_size']]
weights = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i]) for i in range(len(layers) - 1)]
biases = [np.zeros((1, layers[i+1])) for i in range(len(layers) - 1)]

# Optimizer state initialization
m_w = [np.zeros_like(w) for w in weights]  # For Adam
v_w = [np.zeros_like(w) for w in weights]
m_b = [np.zeros_like(b) for b in biases]
v_b = [np.zeros_like(b) for b in biases]
s_w = [np.zeros_like(w) for w in weights]  # For RMSprop
s_b = [np.zeros_like(b) for b in biases]

def adam_update(dW, db, t, idx):
    """
    Updates weights and biases using the Adam optimization algorithm.
    Adam combines momentum and RMSprop for efficient gradient updates.
    """
    global m_w, v_w, m_b, v_b
    beta1 = config['beta1']
    beta2 = config['beta2']
    lr = config['learning_rate']

    # Update biased first moment estimate
    m_w[idx] = beta1 * m_w[idx] + (1 - beta1) * dW
    m_b[idx] = beta1 * m_b[idx] + (1 - beta1) * db

    # Update biased second raw moment estimate
    v_w[idx] = beta2 * v_w[idx] + (1 - beta2) * (dW ** 2)
    v_b[idx] = beta2 * v_b[idx] + (1 - beta2) * (db ** 2)

    # Compute bias-corrected first moment estimate
    m_w_hat = m_w[idx] / (1 - beta1 ** t)
    m_b_hat = m_b[idx] / (1 - beta1 ** t)

    # Compute bias-corrected second raw moment estimate
    v_w_hat = v_w[idx] / (1 - beta2 ** t)
    v_b_hat = v_b[idx] / (1 - beta2 ** t)

    # Update weights and biases
    weights[idx] -= lr * m_w_hat / (np.sqrt(v_w_hat) + config['epsilon'])
    biases[idx] -= lr * m_b_hat / (np.sqrt(v_b_hat) + config['epsilon'])

def rmsprop_update(dW, db, idx):
    """
    Updates weights and biases using the RMSprop optimization algorithm.
    RMSprop scales gradients based on their recent magnitude.
    """
    global s_w, s_b
    decay_rate = config['decay_rate']
    lr = config['learning_rate']

    # Update moving average of squared gradients
    s_w[idx] = decay_rate * s_w[idx] + (1 - decay_rate) * (dW ** 2)
    s_b[idx] = decay_rate * s_b[idx] + (1 - decay_rate) * (db ** 2)

    # Update weights and biases
    weights[idx] -= lr * dW / (np.sqrt(s_w[idx]) + config['epsilon'])
    biases[idx] -= lr * db / (np.sqrt(s_b[idx]) + config['epsilon'])

# Forward pass with optional dropout
def forward(X):
    global activations, pre_activations
    activations = [X]
    pre_activations = []
    for i in range(len(weights) - 1):
        Z = np.dot(activations[-1], weights[i]) + biases[i]
        pre_activations.append(Z)
        act_func = activation_functions[config['activation_functions'][i]][0]
        A = act_func(Z)
        if config['use_dropout']:
            A = apply_dropout(A, config['dropout_rate'])
        activations.append(A)
    Z_final = np.dot(activations[-1], weights[-1]) + biases[-1]
    pre_activations.append(Z_final)
    A_final = softmax(Z_final)
    activations.append(A_final)
    return A_final

# Softmax function
def softmax(x):
    """
    Computes the softmax function for input logits.
    Converts logits to probabilities for multi-class classification.
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Backward pass
def backward(X, y_true, output, t):
    """
    Performs the backward propagation through the neural network.
    Updates weights and biases based on the calculated gradients.
    """
    global weights, biases
    m = y_true.shape[0]
    dZ = output - y_true
    for i in reversed(range(len(weights))):
        dW = np.dot(activations[i].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m

        if config['optimizer'] == 'adam':
            adam_update(dW, db, t, i)
        elif config['optimizer'] == 'rmsprop':
            rmsprop_update(dW, db, i)
        else:
            # Default gradient descent
            weights[i] -= config['learning_rate'] * dW
            biases[i] -= config['learning_rate'] * db

        if i > 0:
            dA = np.dot(dZ, weights[i].T)
            act_derivative = activation_functions[config['activation_functions'][i - 1]][1]
            dZ = dA * act_derivative(pre_activations[i - 1])

# Training loop with early stopping and accuracy/loss tracking
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(config['num_epochs']):
    """
    Iterates through multiple epochs to train the neural network.
    Tracks performance metrics and uses early stopping to avoid overfitting.
    """
    # Shuffle training data
    permutation = np.random.permutation(X_train.shape[0])
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    epoch_train_loss, epoch_train_correct = 0, 0
    for i in range(0, X_train.shape[0], config['batch_size']):
        X_batch = X_train[i:i + config['batch_size']]
        y_batch = y_train[i:i + config['batch_size']]

        # Forward and backward pass
        output = forward(X_batch)
        epoch_train_loss += -np.sum(y_batch * np.log(output + 1e-8)) / y_batch.shape[0]
        epoch_train_correct += np.sum(np.argmax(output, axis=1) == np.argmax(y_batch, axis=1))

        t = epoch * (X_train.shape[0] // config['batch_size']) + (i // config['batch_size']) + 1  # Time step for Adam
        backward(X_batch, y_batch, output, t)

    # Store training loss and accuracy
    train_losses.append(epoch_train_loss / (X_train.shape[0] // config['batch_size']))
    train_accuracies.append(epoch_train_correct / X_train.shape[0])

    # Validation loss and accuracy
    val_output = forward(X_val)
    val_loss = -np.sum(y_val * np.log(val_output + 1e-8)) / y_val.shape[0]
    val_accuracy = np.mean(np.argmax(val_output, axis=1) == np.argmax(y_val, axis=1))
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Train Loss: {train_losses[-1]:.4f}, '
          f'Val Loss: {val_loss:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracy:.4f}')

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= config['patience']:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

# Plotting loss and accuracy curves
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Loss plot
axs[0].plot(range(len(train_losses)), train_losses, label='Training Loss')
axs[0].plot(range(len(val_losses)), val_losses, label='Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training and Validation Loss over Epochs')
axs[0].legend()
axs[0].grid(True)

# Accuracy plot
axs[1].plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
axs[1].plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Training and Validation Accuracy over Epochs')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Evaluation
def predict(X):
    """
    Predicts class labels for input data using the trained model.
    """
    output = forward(X)
    return np.argmax(output, axis=1)

y_pred = predict(X_test)
y_true = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_true)
print(f'Accuracy: {accuracy:.4f}')

def visualize_predictions(X, y_true, y_pred, num_images=16):
    """
    Visualizes a grid of predictions alongside their true labels.
    Highlights correct and incorrect predictions for review.
    """
    plt.figure(figsize=(10, 10))
    grid_size = int(np.sqrt(num_images))
    indices = np.random.choice(X.shape[0], size=num_images, replace=False)  # Randomly select indices
    for i, idx in enumerate(indices):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        img = X[idx].reshape(28, 28)  # Assuming MNIST 28x28 images
        ax.imshow(img, cmap='gray')
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        correct = "Correct" if true_label == pred_label else "Incorrect"
        ax.set_title(f"True: {true_label}\nPred: {pred_label}\n{correct}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize a grid of predictions
visualize_predictions(X_test, y_true, y_pred, num_images=16)

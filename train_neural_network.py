import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

# Function to initialize neural network parameters
def initialize_params(input_size, hidden_size, output_size):
    W1 = np.random.rand(hidden_size, input_size) - 0.5
    b1 = np.random.rand(hidden_size, 1) - 0.5
    W2 = np.random.rand(output_size, hidden_size) - 0.5
    b2 = np.random.rand(output_size, 1) - 0.5
    return W1, b1, W2, b2

# Activation function: Rectified Linear Unit (ReLU)
def ReLU(Z):
    return np.maximum(Z, 0)

# Activation function: Softmax
def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# One-hot encoding for target labels
def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((num_classes, Y.shape[0]))
    one_hot_Y[Y, np.arange(Y.shape[0])] = 1
    return one_hot_Y

# Derivative of ReLU activation
def deriv_ReLU(Z):
    return (Z > 0).astype(int)

# Backpropagation
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y, num_classes=10)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Update parameters with gradient descent
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Function to compute accuracy
def compute_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.shape[0]

# Function to train the neural network
def train_neural_network(X, Y, X_dev, Y_dev, hidden_size, output_size, learning_rate, num_iterations,
                         beta1=0.9, beta2=0.999, epsilon=1e-8, patience=10):
    input_size, m = X.shape
    W1, b1, W2, b2 = initialize_params(input_size, hidden_size, output_size)

    # Initialize Adam optimizer parameters
    m_t = np.zeros_like(W1)
    v_t = np.zeros_like(W1)
    m_t_b = np.zeros_like(b1)
    v_t_b = np.zeros_like(b1)
    m_t_W2 = np.zeros_like(W2)
    v_t_W2 = np.zeros_like(W2)
    m_t_b2 = np.zeros_like(b2)
    v_t_b2 = np.zeros_like(b2)
    
    t = 0  # Time step counter
    best_dev_accuracy = 0.0  # Keep track of the best development set accuracy
    
    training_losses = []  # For tracking training loss over iterations
    development_accuracies = []  # For tracking development accuracy over iterations

    for i in range(num_iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        
        t += 1  # Increment time step
        best_dev_accuracy = 0.0  # Keep track of the best development set accuracy
        no_improvement_count = 0  # Count of iterations with no improvement
        
        # Update Adam optimizer parameters
        m_t = beta1 * m_t + (1 - beta1) * dW1
        v_t = beta2 * v_t + (1 - beta2) * (dW1 ** 2)
        m_t_b = beta1 * m_t_b + (1 - beta1) * db1
        v_t_b = beta2 * v_t_b + (1 - beta2) * (db1 ** 2)
        m_t_W2 = beta1 * m_t_W2 + (1 - beta1) * dW2
        v_t_W2 = beta2 * v_t_W2 + (1 - beta2) * (dW2 ** 2)
        m_t_b2 = beta1 * m_t_b2 + (1 - beta1) * db2
        v_t_b2 = beta2 * v_t_b2 + (1 - beta2) * (db2 ** 2)
        
        # Bias correction
        m_t_hat = m_t / (1 - beta1 ** t)
        v_t_hat = v_t / (1 - beta2 ** t)
        m_t_hat_b = m_t_b / (1 - beta1 ** t)
        v_t_hat_b = v_t_b / (1 - beta2 ** t)
        m_t_hat_W2 = m_t_W2 / (1 - beta1 ** t)
        v_t_hat_W2 = v_t_W2 / (1 - beta2 ** t)
        m_t_hat_b2 = m_t_b2 / (1 - beta1 ** t)
        v_t_hat_b2 = v_t_b2 / (1 - beta2 ** t)
        
        # Update weights and biases
        W1 -= learning_rate * m_t_hat / (np.sqrt(v_t_hat) + epsilon)
        b1 -= learning_rate * m_t_hat_b / (np.sqrt(v_t_hat_b) + epsilon)
        W2 -= learning_rate * m_t_hat_W2 / (np.sqrt(v_t_hat_W2) + epsilon)
        b2 -= learning_rate * m_t_hat_b2 / (np.sqrt(v_t_hat_b2) + epsilon)
        
        # Compute training loss (cross-entropy)
        loss = -(1/m) * np.sum(np.log(A2[Y, np.arange(m)]))
        training_losses.append(loss)
        
        if i % 10 == 0:
            predictions = np.argmax(A2, axis=0)
            accuracy = compute_accuracy(predictions, Y)
            Z1_dev, A1_dev, Z2_dev, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
            dev_predictions = np.argmax(A2_dev, axis=0)
            development_accuracy = compute_accuracy(dev_predictions, Y_dev)
            print(f"Iteration: {i}, Training Loss: {loss:.4f}, Training Accuracy: {accuracy:.4f}, Development Accuracy: {development_accuracy:.4f}")
            
            # Append development accuracy to the list
            development_accuracies.append(development_accuracy)

            # Check for early stopping based on development set accuracy
            if development_accuracy > best_dev_accuracy:
                best_dev_accuracy = development_accuracy
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Early stopping triggered after {patience} iterations with no improvement.")
                    break
    
    return W1, b1, W2, b2, training_losses, development_accuracies

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)
    
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.0
    
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.0
    
    return X_train, Y_train, X_dev, Y_dev

if __name__ == "__main__":
    # Load and preprocess data
    X_train, Y_train, X_dev, Y_dev = load_and_preprocess_data("mnist_train.csv")
    
    # Train the neural network
    hidden_size = 128
    output_size = 10
    learning_rate = 0.001
    num_iterations = 1000
    early_stopping_patience = 10
    
    W1, b1, W2, b2, training_losses, development_accuracies = train_neural_network(X_train, Y_train, X_dev, Y_dev, hidden_size, output_size, learning_rate, num_iterations, patience=early_stopping_patience)
    
    # Test the trained model on the development set
    dev_predictions = np.argmax(forward_prop(W1, b1, W2, b2, X_dev)[-1], axis=0)
    accuracy = compute_accuracy(dev_predictions, Y_dev)
    print(f"Development Set Accuracy: {accuracy:.4f}")

    # Save the trained model using pickle
    trained_model = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    with open('trained_model.pkl', 'wb') as model_file:
        pickle.dump(trained_model, model_file)

    print("Trained model saved successfully.")

    # Plot training loss and development accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(development_accuracies)
    plt.title("Development Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    
    plt.show()
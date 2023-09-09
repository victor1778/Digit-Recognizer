import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    trained_model = pickle.load(model_file)

W1 = trained_model['W1']
b1 = trained_model['b1']
W2 = trained_model['W2']
b2 = trained_model['b2']

# Load and preprocess the test data
def load_and_preprocess_test_data(file_path):
    data = pd.read_csv(file_path)
    data = np.array(data)
    m, n = data.shape
    
    data_test = data.T
    Y_test = data_test[0]
    X_test = data_test[1:n]
    X_test = X_test / 255.0
    
    return X_test, Y_test

X_test, Y_test = load_and_preprocess_test_data("mnist_test.csv")

# Activation function: Rectified Linear Unit (ReLU)
def ReLU(Z):
    return np.maximum(Z, 0)

# Activation function: Softmax
def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

# Define the forward propagation function
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return A2

# Make predictions on the test data
predictions = np.argmax(forward_prop(W1, b1, W2, b2, X_test), axis=0)

# Compute accuracy
accuracy = np.sum(predictions == Y_test) / Y_test.shape[0]
print(f"Test Set Accuracy: {accuracy:.4f}")

# Visualize sample images and their predicted labels
num_samples_to_visualize = 5

for i in range(num_samples_to_visualize):
    sample_index = np.random.randint(0, X_test.shape[1])
    sample_image = X_test[:, sample_index].reshape(28, 28) * 255
    sample_label = Y_test[sample_index]
    sample_prediction = predictions[sample_index]

    plt.figure()
    plt.title(f"Sample {i + 1}\nTrue Label: {sample_label}, Predicted Label: {sample_prediction}")
    plt.imshow(sample_image, cmap='gray')
    plt.axis('off')

plt.show()
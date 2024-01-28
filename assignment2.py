import numpy as np

# Define the input and output arrays for the XOR function
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Initialize the weights and biases to random values
weights = np.random.rand(6)
bias = np.random.rand(3)

# Define the activation function for the perceptron
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the loss function for the perceptron
def loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Define the gradient descent function to update the weights and biases
def gradient_descent(X, y, weights, bias, learning_rate, epochs):
    for epoch in range(epochs):
        # Forward pass
        v1 = sigmoid(-weights[0]*X[:,0] - weights[1]*X[:,1] + bias[0]) # Check if both are 0
        v2 = sigmoid(weights[2]*X[:,0] + weights[3]*X[:,1] + bias[1]) # Check if both are 1
        z = sigmoid(-weights[4]*v1 - weights[5]*v2 + bias[2]) # Check that neither are true
        y_pred = z

        # Backward pass
        error = y - y_pred
        d_weights = np.zeros(6)
        d_bias = np.zeros(3)
        for i in range(4):
            d_weights[0] += error[i] * v1[i] * (1 - v1[i]) * X[i,0]
            d_weights[1] += error[i] * v1[i] * (1 - v1[i]) * X[i,1]
            d_weights[2] += error[i] * v2[i] * (1 - v2[i]) * X[i,0]
            d_weights[3] += error[i] * v2[i] * (1 - v2[i]) * X[i,1]
            d_weights[4] += error[i] * z[i] * (1 - z[i]) * v1[i]
            d_weights[5] += error[i] * z[i] * (1 - z[i]) * v2[i]
            d_bias[0] += error[i] * v1[i] * (1 - v1[i])
            d_bias[1] += error[i] * v2[i] * (1 - v2[i])
            d_bias[2] += error[i] * z[i] * (1 - z[i])

        # Update weights and biases
        weights += learning_rate * d_weights
        bias += learning_rate * d_bias

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss(y, y_pred)}")

    return weights, bias

# Train the perceptron using the gradient descent function
weights, bias = gradient_descent(X, y, weights, bias, learning_rate=0.01, epochs=10000)

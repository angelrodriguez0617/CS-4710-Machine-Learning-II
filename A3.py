import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np

# Test learning rates from 10-100
learning_rates = np.arange(10, 110, 10)

# Initialize an empty dictionary to keep track of learning rates with corresponding losses
learning_loss_dict = {}

# Training data, inputs and outputs for an XOR gate
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

for rate in learning_rates:
    print(f'Training model with learning rate = {rate}')
    # Build the model
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(2,)),  # Hidden layer with 4 neurons
        Dense(1, activation='sigmoid')                    # Output layer
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=rate), loss='mse')

    # Define a callback to print results every 10 epochs
    print_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch + 1}/{1000} - Loss: {logs['loss']}")
        if (epoch + 1) % 100 == 0 else None
    )

    # Train the model with reduced verbosity and the callback
    model.fit(x_train, y_train, epochs=1000, callbacks=[print_callback], verbose=0)

    # Print the learned weights and biases
    print("\nLearned Weights:")
    for layer in model.layers:
        print(layer.get_weights()[0])

    print("\nLearned Biases:")
    for layer in model.layers:
        print(layer.get_weights()[1])

    # Print the function the model uses to make predictions
    def predict_function(x):
        for i, layer in enumerate(model.layers):
            x = np.dot(x, layer.get_weights()[0]) + layer.get_weights()[1]
            if i < len(model.layers) - 1:
                x = 1 / (1 + np.exp(-x))  # Apply sigmoid activation for hidden layers
        return x

   # Use the function to make predictions
    print("\nPredictions:")
    for input_data in x_train:
        raw_prediction = predict_function(input_data)
        sigmoid_prediction = 1 / (1 + np.exp(-raw_prediction))
        print(f"Input: {input_data}, Raw Prediction: {raw_prediction}, Sigmoid Prediction: {sigmoid_prediction}")

    # Evaluate the model on the training data
    loss = model.evaluate(x_train, y_train)
    learning_loss_dict[rate] = loss
    print(f'Final Loss: {loss}')
    print('\n--------------------------------------------------------------------------------------------\n')

min_key = min(learning_loss_dict, key=learning_loss_dict.get)
print(f'The minimum loss is {learning_loss_dict[min_key]} which came from a learning rate of {min_key}\n')  

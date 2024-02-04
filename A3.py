import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np

learning_rates = np.arange(10, 110, 10)

# Initialize an empty dictionary
learning_loss_dict = {}

# Training data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

for rate in learning_rates:

    # Build the model
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(2,)),  # Hidden layer with 4 neurons
        Dense(1, activation='sigmoid')                    # Output layer
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=rate), loss='mse')

    # Define a callback to print results every 10 epochs
    print_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch + 1}/{1000} - Loss: {logs['loss']}, learning rate = {rate}")
        if (epoch + 1) % 100 == 0 else None
    )

    # Train the model with reduced verbosity and the callback
    model.fit(x_train, y_train, epochs=1000, callbacks=[print_callback], verbose=0)

    # Evaluate the model on the training data
    loss = model.evaluate(x_train, y_train)
    learning_loss_dict[rate] = loss
    print(f'Final Loss: {loss}')

min_key = min(learning_loss_dict, key=learning_loss_dict.get)
print(f'The minimum loss is {learning_loss_dict[min_key]} which came from a learning rate of {min_key}')  

# CS6480 Assignment #2
# Code from scratch. Derive equations and implement the XOR(x1, x2), where x1 and x2 take values of {0, 1}.
import random # Only for generating random integers
import numpy as np

acceptable_loss = 0.01

def populate_array(n):
    arr = []
    for i in range(n):
        # arr.append(random.choice([-1, 1]) * random.randint(1, 10))
        arr.append(random.choice([-1, 1]) * random.randint(1, 10))
    return arr

def exp(x):
    '''Calculate the exponential function of x'''
    result = 1
    term = 1
    for n in range(1, 100):
        term *= x / n
        result += term
    return result

def sigmoid(x):
    '''Calculate the sigmoid of x'''
    return 1 / (1 + exp(-x))

def sigmoid_derivative(x):
    """
    This function returns the derivative of the sigmoid function.
    """
    my_sigmoid = sigmoid(x)
    return my_sigmoid * (1 - my_sigmoid)

def XOR(x1, x2, w, b):
    '''Perform XOR function using perceptrons'''

    v1 = sigmoid(-w[0]*x1 - w[1]*x2 + b[0]) # Check if both are 0
    v2 = sigmoid(w[2]*x1 + w[3]*x2 + b[1]) # Check if both are 1

    # z is 1 if x1 and x2 are different (like XOR gate)
    z = sigmoid(-w[4]*v1 - w[5]*v2 + b[2]) # Check that neither are true
    return z

def calc_loss(x1_array, x2_array, w, b):
    loss = 0
    for x1, x2 in zip(x1_array, x2_array):
        # print(f'x1, x2 = {x1}, {x2}')
        if x1 != x2:
            expected_z = 1
        else:
            expected_z = 0
        z = XOR(x1, x2, w, b)
        # print(f'expected z = {expected_z}, actual z = {z}')
        if z == expected_z:
            # print('actual output is expected output\n')
            pass
        else:
            # print('actual output is not expected output\n')
            pass
        loss += (expected_z - z)**2
    # print(f'loss {loss}\n')
    if loss < acceptable_loss:
        print(f'Loss is at acceptable value of {loss}')
    return loss
   
def backprop(x1_array, x2_array, w, b, learning_rate):

    # Update the values of w_array and b_array
    for i in range(len(w)):
        # w[i] -= learning_rate * dw[i]
        orignal_variable = w[i]
        original_loss = calc_loss(x1_array, x2_array, w, b)
        if original_loss < acceptable_loss:
            break
        while not isinstance(w[i], (int, float, complex)) or w[i] - orignal_variable == 0 or w[i] == 0:
            w[i] = random.choice([-1, 1]) * random.randint(1, 10) # random.randrange(2,10)
        new_loss = calc_loss(x1_array, x2_array, w, b)
        if new_loss < acceptable_loss:
            break
        loss_derivative = (new_loss - original_loss) / (w[i] - orignal_variable)
        # print(f'loss_derivative: {loss_derivative}')
        if loss_derivative == 0 or not isinstance(loss_derivative, (int, float, complex)) :
            w[i] = orignal_variable
        else:
            w[i] = orignal_variable - (learning_rate * loss_derivative)
    for i in range(len(b)):
        # b[i] -= learning_rate * db[i]
        orignal_variable = b[i]
        original_loss = calc_loss(x1_array, x2_array, w, b)
        if original_loss < acceptable_loss:
            break
        while not isinstance(w[i], (int, float, complex)) or b[i] - orignal_variable == 0 or b[i] == 0:
            b[i] = random.choice([-1, 1]) * random.randint(1, 10) # random.randrange(2,10)
        new_loss = calc_loss(x1_array, x2_array, w, b)
        if new_loss < acceptable_loss:
            break
        loss_derivative = (new_loss - original_loss) / (b[i] - orignal_variable)
        # print(f'loss_derivative: {loss_derivative}')
        if loss_derivative == 0 or not isinstance(loss_derivative, (int, float, complex)):
            b[i] = orignal_variable
        else:
            b[i] = orignal_variable - (learning_rate * loss_derivative)

    final_loss = calc_loss(x1_array, x2_array, w, b)
    return final_loss, w, b

# Create a vector for weights
w_array = populate_array(6)
# Create a vector for constants
b_array = populate_array(3)

# Define learning rate
learning_rate = 1

x1_inputs = [0, 1, 0, 1]
x2_inputs = [0, 0, 1, 1]

first_w = w_array.copy()
first_b = b_array.copy()

losses = []
first_loss = calc_loss(x1_inputs, x2_inputs, w_array, b_array)
if first_loss < acceptable_loss:
    print(f'Updated w_array: {w_array}')
    print(f'Updated b_array: {b_array}')
    print(f'Final loss: {first_loss}')
else:
    # Perform backpropagation for 1000 iterations
    for i in range(1000):
        loss, w_array, b_array = backprop(x1_inputs, x2_inputs, w_array, b_array, learning_rate)
        if loss < acceptable_loss:
            break
        losses.append(loss)

    loss = min(losses)

    print(f'Initial w_array: {first_w}')
    print(f'Initial b_array: {first_b}')
    print(f'Updated w_array: {w_array}')
    print(f'Updated b_array: {b_array}')
    print(f'Initial loss: {first_loss}')
    print(f'Final loss: {loss}')
    print(f'Error reduced by: {round(100 * (first_loss - loss) / first_loss)}%')






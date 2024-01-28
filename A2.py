# CS6480 Assignment #2
# Code from scratch. Derive equations and implement the XOR(x1, x2), where x1 and x2 take values of {0, 1}.

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
        print(f'x1, x2 = {x1}, {x2}')
        if sigmoid(x1) != sigmoid(x2):
            expected_z = 1
        else:
            expected_z = 0
        z = XOR(x1, x2, w, b)
        print(f'expected z = {expected_z}, actual z = {z}')
        if z == expected_z:
            # print('actual output is expected output\n')
            pass
        else:
            # print('actual output is not expected output\n')
            pass
        loss += (expected_z - z)**2
    loss /= 2
    print(f'loss {loss}\n')
    return loss
   
def backprop(x1_array, x2_array, w, b, learning_rate):
    # Calculate the output of the XOR function for each input pair
    outputs = []
    for x1, x2 in zip(x1_array, x2_array):
        outputs.append(XOR(x1, x2, w, b))

    # Calculate the loss function
    loss = calc_loss(x1_array, x2_array, w, b)

    # Calculate the partial derivatives of the loss function
    dw = [0] * len(w)
    db = [0] * len(b)
    for i in range(len(x1_array)):

        v1 = sigmoid(-w[0]*x1 - w[1]*x2 + b[0]) # Check if both are 0
        v2 = sigmoid(w[2]*x1 + w[3]*x2 + b[1]) # Check if both are 1

        z = sigmoid(-w[4]*v1 - w[5]*v2 + b[2])

        if x1_array[i] != x2_array[i]:
            expected_z = 1
        else:
            expected_z = 0

        dw[0] += (expected_z - z) * z * (1 - z) * v1 
        dw[1] += (expected_z - z) * z * (1 - z) * v1
        dw[2] += (expected_z - z) * z * (1 - z) * v2 
        dw[3] += (expected_z - z) * z * (1 - z) * v2 
        dw[4] += (expected_z - z) * v1 * z * (1 - z) * w[5]
        dw[5] += (expected_z - z) * v2 * z * (1 - z) * w[4]
        db[0] += (expected_z - z) * z * (1 - z) * v1
        db[1] += (expected_z - z) * z * (1 - z) * v2
        db[2] += (expected_z - z) * z * (1 - z)

    # Update the values of w_array and b_array
    for i in range(len(w)):
        w[i] -= learning_rate * dw[i]
    for i in range(len(b)):
        b[i] -= learning_rate * db[i]

    return loss, w, b

# Create a vector for weights
w_array = [1, 1, 1, 1, 1, 1]
# Create a vector for constants
b_array = [0.5, -1.5, 0.5]
# Learning rate
learning_rate = 0.01

x1_inputs = [0, 1, 0, 1]
x2_inputs = [0, 0, 1, 1]

first_loss = calc_loss(x1_inputs, x2_inputs, w_array, b_array)
# Perform backpropagation for 1000 iterations
for i in range(1000):
    loss, w_array, b_array = backprop(x1_inputs, x2_inputs, w_array, b_array, learning_rate)

print(f'Initial loss: {first_loss}')
print(f'Final loss: {loss}')
print(f'Updated w_array: {w_array}')
print(f'Updated b_array: {b_array}')






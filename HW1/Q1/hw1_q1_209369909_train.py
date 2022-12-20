from keras.datasets import mnist
import numpy as np
import pickle
import matplotlib.pyplot as plt


def flattening(nump_arr):
    flattened = []
    for mat in nump_arr:
        flattened.append(mat.flatten(order='C'))
    return np.array(flattened)


def tanh(x):
    return np.tanh(x)

def softmax(x):
    expX = np.exp(x)
    return expX/np.sum(expX, axis = 0)



def derivative_relu(x):
    return np.array(x > 0, dtype = np.float32)

def derivative_tanh(x):
    return (1 - np.power(np.tanh(x), 2))


def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))

    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }

    return parameters


def forward_propagation(x, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, x) + b1
    a1 = tanh(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)

    forward_cache = {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2
    }

    return forward_cache


def cost_function(a2, y):
    return -(1 / y.shape[1]) * np.sum(y * np.log(a2))


def backward_prop(x, y, parameters, forward_cache):
    w2 = parameters['w2']
    a1 = forward_cache['a1']
    a2 = forward_cache['a2']

    m = x.shape[1]

    dz2 = (a2 - y)
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

    dz1 = (1 / m) * np.dot(w2.T, dz2) * derivative_tanh(a1)
    dw1 = (1 / m) * np.dot(dz1, x.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    gradients = {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2
    }

    return gradients


def update_parameters(parameters, gradients, learning_rate):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = gradients['dw1']
    db1 = gradients['db1']
    dw2 = gradients['dw2']
    db2 = gradients['db2']

    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2

    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }

    return parameters


def model(x, y, n_h, learning_rate, iterations):
    n_x = x.shape[0]
    n_y = y.shape[0]

    cost_list = []

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(iterations):
        forward_cache = forward_propagation(x, parameters)
        cost = cost_function(forward_cache['a2'], y)
        gradients = backward_prop(x, y, parameters, forward_cache)
        parameters = update_parameters(parameters, gradients, learning_rate)
        cost_list.append(cost)

        if (i % 10 == 0):
            print("Cost after", i, "iterations is :", cost)

    return parameters, cost_list


def accuracy(inp, labels, parameters):
    forward_cache = forward_propagation(inp, parameters)
    a_out = forward_cache['a2']  # containes propabilities with shape(10, 1)
    a_out = np.argmax(a_out, 0)  # 0 represents row wise
    labels = np.argmax(labels, 0)
    acc = np.mean(a_out == labels) * 100
    return acc


if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = flattening(train_X).T
    train_y_mat = (train_y[:, None] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(int)
    # adapting types
    train_y = train_y_mat.T
    test_X = flattening(test_X).T
    test_y_mat = (test_y[:, None] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(int)
    test_y = test_y_mat.T


    iterations = 100
    n_h = 1000
    learning_rate = 0.01
    Parameters, Cost_list = model(train_X, train_y, n_h, learning_rate, iterations)


    file_name='model.pkl'
    f = open(file_name,'wb')
    pickle.dump(Parameters,f)
    f.close()

    print("Accuracy of Train Dataset", accuracy(train_X, train_y, Parameters), "%")
    print("Accuracy of Test Dataset", round(accuracy(test_X, test_y, Parameters), 2), "%")
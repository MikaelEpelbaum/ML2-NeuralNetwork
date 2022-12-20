import torch
import torchvision.datasets as dsets
from torchvision import transforms
import pickle
import numpy as np
from hw1_q1_209369909_train import forward_propagation


def make_predictions(X, parameters,):# W1, b1, W2, b2):
    A2 = forward_propagation(X, parameters)['a2']
    predictions = np.argmax(np.where(A2 > 0, A2, 0), axis=0)
    return predictions

def evaluate_hw1_q1():
    # Load Model
    data = []
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)

    # Load test data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.3081, ], std=[0.1306, ])])

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               download=True,
                               transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=10000,
                                              shuffle=False)

    correct = 0
    total = 0
    for X_test, Y_test in test_loader:
        X_test = X_test.view(-1, 28 * 28)
        X_test = X_test.T

        predictions = make_predictions(X_test.float(), data)

        total += Y_test.size(0)
        correct += (predictions == Y_test.numpy()).sum()

    accuracy = (float(correct) / total) * 100
    print("Question 1, Test accuracy =", accuracy)
    return accuracy


if __name__ == "__main__":
    evaluate_hw1_q1()

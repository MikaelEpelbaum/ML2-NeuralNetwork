from torchvision import datasets, transforms
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torch
import matplotlib.pyplot as plt

Glob_images = None

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 28*28, 100
        self.fc1 = nn.Linear(784, 100)

        # hidden layers
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)

        # output layer
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)



def train_model(iterations, network, classes, train_data):
    # training the neural network with the dataset for x iterations
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    for first_batch in train_data:
        images, classes = first_batch
        break

    if iterations == 0:
        loss = F.cross_entropy(images.view(-1, 784), classes)
    else:
        for time in range(iterations):
            network.zero_grad()
            output = network(images.view(-1, 784))
            loss = F.cross_entropy(output, classes)
            loss.backward()
            optimizer.step()

    return loss


def get_domain(size=None):
    train = datasets.MNIST("", train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))

    test = datasets.MNIST("", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train, batch_size=size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=size, shuffle=False)

    return train_loader, test_loader


def loss_calc():
    trainLoss = []
    testLoss = []
    initial_class = np.random.binomial(size=128, n=1, p=0.5)
    for i in range(301):
        if i%50 == 0:
            print('epoch: ' + str(i))

        network = NeuralNetwork()
        loss = train_model(i, network, initial_class, train_data)
        trainLoss.append(loss.item())

        for first_batch in train_data:
            images, classes = first_batch
            break
        tempLoss = F.cross_entropy(network(images.view(-1, 784)), torch.from_numpy(initial_class).type(torch.LongTensor))
        testLoss.append(tempLoss.item())
    return network, trainLoss, testLoss



if __name__ == '__main__':
    train_data, test = get_domain(128)
    network, trainLoss, testLoss = loss_calc()

    print("The loss on the Train data is : " + str(trainLoss[300]))
    print("The loss on the Test data is : " + str(testLoss[300]))

    # Save the Model
    torch.save(network.state_dict(), 'model2.pkl')

    # Plot the train accuracy per epochs
    plt.plot(trainLoss, label='Train loss.')
    plt.title('Train loss per epochs on MNIST')
    plt.ylabel('Accuracy in %')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot the train accuracy per epochs
    plt.plot(testLoss, label='Test loss.')
    plt.title('Test loss per epochs on MNIST')
    plt.ylabel('Accuracy in %')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()
    quit()
# Import necessary libraries
import torch
from torch.nn import functional as F
import numpy as np
from hw1_q2_209369909_train import get_domain, NeuralNetwork


if __name__ == "__main__":
    # Get MNIST Dataset (Images and Labels)
    train_loader, test_loader = get_domain(size=128)

    # Load saved torch model
    model = NeuralNetwork()
    model.load_state_dict(torch.load('model2.pkl'))
    model.eval()

    # Get random labels as in train:
    random_labels = torch.from_numpy(np.random.binomial(size=128, n=1, p=0.5))

    # Variables to store our loss per epoch
    lossTrain = []
    lossTest = []

    # Run inference on the train model
    for first_batch in train_loader:
        images, labels = first_batch
        break

    train_output = model(images.view(-1, 28 * 28))
    lossTrain = F.cross_entropy(train_output, random_labels.type(torch.LongTensor))

    # Run inference on the test model
    for first_batch in test_loader:
        images, labels = first_batch
        break
    test_output = model(images.view(-1, 28 * 28))
    lossTest = F.cross_entropy(test_output, random_labels.type(torch.LongTensor))

    print("The loss on the Train data is : " + str(lossTrain.item()))
    print("The loss on the Test data is : " + str(lossTest.item()))







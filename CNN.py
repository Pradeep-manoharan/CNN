# Setup & Library

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

# Device Configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-Parameters

num_inputs = 28 * 28
num_epochs = 1
batch_size = 4
learning_rate = 0.01

# Dataset Preparation

# dataset has PIL Image of range(0,1)
# We transform them to tensor of normalize range of [-1,1]

transform = transform.Compose([transform.ToTensor(), transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root="\data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="\data", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5  # unNormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Get some random image
example = iter(train_loader)
image, label = next(example)


# show the image
# imshow(torchvision.utils.make_grid(image))


# Model Building

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = CNN().to(device)

# Training

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

n_total_step = len(train_loader)

for epochs in range(num_epochs):
    for i, (image, label) in enumerate(train_loader):
        # original shape = (4,3,32,32) = 4,3,1024
        # Input layer : Input_Channel : 3, Output_Channel : 6, Kernel_size :5

        image = image.to(device)
        label = label.to(device)

        # forward pass
        output = model(image)
        loss = criteria(output, label)

        # backward propagation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epochs [{epochs + 1}/{num_epochs}],[{i + 1}/{n_total_step}],Loss:[{loss.item()}]")

print("Finished Training")

with torch.no_grad():
    n_correct = 0
    n_sample = 0

    n_class_correct = [0 for i in range(10)]
    n_class_sample = [0 for i in range(10)]

    for image, labels in test_loader:
        image = image.to(device)
        labels = labels.to(device)

        output = model(image)

        # Max Return (value, index)

        _, predicted = torch.max(output, 1)
        n_sample += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_sample[label] += 1

    accuracy = 100.0 * n_correct / n_sample
    print(f'Accuracy of the network: {accuracy} %')

    for i in range(10):
        accuracy = 100.0 * n_class_correct[i] / n_class_sample[i]
        print(f'Accuracy of {classes[i]}: {accuracy} %')

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])

transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
train_ds = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=100, shuffle=True)

validation_ds = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
validation_loader = torch.utils.data.DataLoader(dataset=validation_ds, batch_size=100)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(4*4*64, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)

        self.training_loss_history = []
        self.training_corrects_history = []
        self.validation_loss_history = []
        self.validation_corrects_history = []

    def forward(self, x, training=True):
        pred = F.relu(self.conv1(x))
        pred = F.max_pool2d(pred, 2, 2)
        pred = F.relu(self.conv2(pred))
        pred = F.max_pool2d(pred, 2, 2)
        pred = F.relu(self.conv3(pred))
        pred = F.max_pool2d(pred, 2, 2)
        pred = pred.view(-1, 4*4*64)
        pred = F.relu(self.fc1(pred))
        if training:
            pred = self.dropout1(pred)
        pred = self.fc2(pred)
        return pred

    def learn(self, epochs=16, validate=False):
        self.training_loss_history = []
        self.training_corrects_history = []
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

        for e in range(epochs):
            running_loss = 0.0
            running_corrects = 0.0
            for inputs, labels in train_loader:
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                running_loss += loss.item()
            else:
                epoch_loss = running_loss/len(train_loader)
                epoch_accuracy = running_corrects.float()/len(train_loader)
                self.training_loss_history.append(epoch_loss)
                self.training_corrects_history.append(epoch_accuracy)
                print(f'training loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.4f}')
            if validate:
                self.validate(validation_loader, True)

    def validate(self, loader, track_loss_and_accuracy=False):
        criterion = nn.CrossEntropyLoss()
        running_corrects = 0.0
        running_loss = 0.0

        for inputs, labels in loader:
            outputs = self.forward(inputs, False)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).float()
            running_loss += loss.item()
        else:
            epoch_accuracy = running_corrects / len(loader)
            epoch_loss = running_loss / len(loader)
            if track_loss_and_accuracy:
                self.validation_corrects_history.append(epoch_accuracy)
                self.validation_loss_history.append(epoch_loss)
            print(f'validation loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.4f}')
        return epoch_loss


model = LeNet()
model_file = 'cf10_lenet_state_dict.txt'
model_ready = False

if os.path.isfile(model_file):
    if 'y' == input('load existing state? (y/n)').lower():
        model.load_state_dict(torch.load(model_file))
        model_ready = True

if not model_ready:
    model.learn(validate=True)
    if 'y' == input('save trained state? (y/n)').lower():
        torch.save(model.state_dict(), model_file)
    model_ready = True
#print(model)

model.validate(validation_loader)

# plt.plot(model.training_loss_history, label='training loss')
# plt.plot(model.validation_loss_history, label='validation loss')
# plt.legend()
#
# plt.plot(model.training_corrects_history, label='training accuracy')
# plt.plot(model.validation_corrects_history, label='validation accuracy')
# plt.legend()




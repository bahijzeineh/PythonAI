import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os


def im_convert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
    image = image.clip(0, 1)
    return image


transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])

train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=100, shuffle=True)

validation_ds = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
validation_loader = torch.utils.data.DataLoader(dataset=validation_ds, batch_size=100)

# data_iter = iter(train_loader)
# images, labels = data_iter.next()
# fig = plt.figure(figsize=(25,4))
# for idx in np.arange(20):
#     fig.add_subplot(2,10,idx+1)
#     plt.imshow(im_convert(images[idx]))
# plt.show()


class Classifier(nn.Module):
    def __init__(self, input_size, h1, h2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, output_size)

        self.running_loss_history = []
        self.running_corrects_history = []

    def forward(self, x):
        pred = F.relu(self.linear1(x))
        pred = F.relu(self.linear2(pred))
        pred = self.linear3(pred) # for classifiers dont apply activation function on final layer
        return pred

    def learn(self):
        self.running_loss_history = []
        self.running_corrects_history = []
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
        epochs = 16
        for e in range(epochs):
            running_loss = 0.0
            running_corrects = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.view(inputs.shape[0], -1)
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
                self.running_loss_history.append(epoch_loss)
                self.running_corrects_history.append(epoch_accuracy)
                print(f'loss: {epoch_loss:.4f}')
                print(f'accuracy: {epoch_accuracy:.4f}')

    def validate(self, loader):
        running_corrects = 0.0
        for inputs, labels in loader:
            inputs = inputs.view(inputs.shape[0], -1)
            outputs = self.forward(inputs)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        else:
            epoch_accuracy = running_corrects.float() / len(loader)
            print(f'validation accuracy: {epoch_accuracy:.4f}')


model = Classifier(784, 125, 65, 10)
model_file = 'ir_state_dict.txt'
model_ready = False

if os.path.isfile(model_file):
    choice = input('load existing state? (y/n)')
    if choice.lower() == 'y':
        model.load_state_dict(torch.load(model_file))
        model_ready = True

if not model_ready:
    model.learn()
    if 'y' == input('save trained state? (y/n)').lower():
        torch.save(model.state_dict(), model_file)
    model_ready = True
#print(model)

model.validate(validation_loader)

# plt.plot(model.running_loss_history)
# plt.plot(model.running_corrects_history)



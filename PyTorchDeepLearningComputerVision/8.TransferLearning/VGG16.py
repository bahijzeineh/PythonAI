import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
from PIL import Image


transform_train = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                      transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
train_ds = datasets.ImageFolder('ants_and_bees/train', transform=transform_train)
train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=20, shuffle=True)

validation_ds = datasets.ImageFolder('ants_and_bees/val', transform=transform)
validation_loader = torch.utils.data.DataLoader(dataset=validation_ds, batch_size=20)

classes = ['ant', 'bee']

model = models.vgg16(False)
model.load_state_dict(torch.load('vgg16-397923af.pth'))

for param in model.features.parameters():
    param.requires_grad = False
n_inputs = model.classifier[6].in_features
last_layer = nn.Linear(n_inputs,2)
model.classifier[6] = last_layer

training_loss_history = []
training_corrects_history = []
validation_corrects_history = []
validation_loss_history = []


def learn(epochs=5, validate=False):
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    for e in range(epochs):
        running_loss = 0.0
        running_corrects = 0.0
        for inputs, labels in train_loader:
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            running_loss += loss.item()
        else:
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_accuracy = running_corrects.float() / len(train_loader.dataset)
            training_loss_history.append(epoch_loss)
            training_corrects_history.append(epoch_accuracy)
            print(f'training loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.4f}')
        if validate:
            validation(validation_loader, True)


def validation(loader, track_loss_and_accuracy=False):
    criterion = nn.CrossEntropyLoss()
    running_corrects = 0.0
    running_loss = 0.0

    for inputs, labels in loader:
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).float()
        running_loss += loss.item()
    else:
        epoch_accuracy = running_corrects / len(loader.dataset)
        epoch_loss = running_loss / len(loader.dataset)
        if track_loss_and_accuracy:
            validation_corrects_history.append(epoch_accuracy)
            validation_loss_history.append(epoch_loss)
        print(f'validation loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.4f}')
    return epoch_loss


model_file = 'ab_vgg16_state_dict.txt'
model_ready = False

if os.path.isfile(model_file):
    if 'y' == input('load existing state? (y/n)').lower():
        model.load_state_dict(torch.load(model_file))
        model_ready = True

if not model_ready:
    learn(validate=True)
    if 'y' == input('save trained state? (y/n)').lower():
        torch.save(model.state_dict(), model_file)
    model_ready = True
#print(model)

# model.validate(validation_loader)

# plt.plot(model.training_loss_history, label='training loss')
# plt.plot(model.validation_loss_history, label='validation loss')
# plt.legend()
#
# plt.plot(model.training_corrects_history, label='training accuracy')
# plt.plot(model.validation_corrects_history, label='validation accuracy')
# plt.legend()


def im_convert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image


def predict(url):
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)
    img = transform(img)
    image = img.unsqueeze(0)
    output = model(image)
    _, pred = torch.max(output, 1)
    return classes[pred.item()]

bee_url1 = 'https://images.squarespace-cdn.com/content/v1/5bca52142727be691c3bdbb9/1543379385336-DNQG9KKFP18RD9OX474R/ke17ZwdGBToddI8pDm48kC6bbDusiSGD6dxPOM9U5m5Zw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZamWLI2zvYWH8K3-s_4yszcp2ryTI0HqTOaaUohrI8PI3aFgoaGOuahUK_emfSZZiDzsv8k_Y3rRG-HLtWSKx4IKMshLAGzx4R3EDFOm1kBS/1+Large+Bee.jpg'
bee_url2 = 'https://assets.newatlas.com/dims4/default/7d91e73/2147483647/strip/true/crop/1583x1056+0+12/resize/1160x774!/quality/90/?url=https%3A%2F%2Fassets.newatlas.com%2Farchive%2Fwallaces-giant-bee-2.jpg'
bee_url3 = 'https://cdn.vox-cdn.com/thumbor/TdjLfYgyy54QNDooQguJSjRXD4s=/0x243:2500x2118/1200x800/filters:focal(0x243:2500x2118)/cdn.vox-cdn.com/uploads/chorus_image/image/46679984/shutterstock_150559442.0.0.jpg'
ant_url1 = 'https://images.newscientist.com/wp-content/uploads/2019/10/16151942/cataglyphisbombycinasoldier1uniulmdouzfotohwolf.jpg'
ant_url2 = 'https://images.ctfassets.net/cnu0m8re1exe/2rRUQeM5ru2OOFKksT60GC/4763205926d05b29936959afb4ff6723/big_ant.jpg'
ant_url3 = 'https://static01.nyt.com/images/2014/05/27/science/27take/27take-jumbo.jpg'

print(predict(bee_url1))
print(predict(bee_url2))
print(predict(bee_url3))
print(predict(ant_url1))
print(predict(ant_url2))
print(predict(ant_url3))
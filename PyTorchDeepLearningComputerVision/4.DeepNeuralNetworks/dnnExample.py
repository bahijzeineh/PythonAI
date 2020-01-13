import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

n_pts = 500
X, y = datasets.make_circles(n_samples=n_pts, noise = 0.1, factor=0.2, random_state=123)
x_data = torch.FloatTensor(X)
y_data = torch.FloatTensor(y).reshape(n_pts,1 )


class Model(nn.Module):
    def __init__(self, input_size, h1, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Linear(input_size, h1)
        self.linear2 = nn.Linear(h1, output_size)

    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))
        pred = torch.sigmoid(self.linear2(pred))
        return pred

    def get_params(self):
        [wts, b] = model.parameters()
        params = [w.item() for w in wts.view(self.input_size)]
        b = b[0].item()
        params.append(b)
        return params

    def learn(self, epochs=1000):
        criterion = nn.BCELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.1)

        for i in range(epochs):
            y_pred = model.forward(x_data)
            loss = criterion(y_pred, y_data)
            print(f'epoch: {i}, loss: {loss:.4f}')
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    def plot_decision_boundary(self, X, y):
        x_span = np.linspace(min(X[:, 0]), max(X[:, 0]))
        y_span = np.linspace(min(X[:, 1]), max(X[:, 1]))

        xx, yy = np.meshgrid(x_span, y_span)
        grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        pred_func = self.forward(grid)
        z = pred_func.view(xx.shape).detach().numpy()
        plt.contourf(xx, yy, z)

    def plot_fit(self, title='DNN Model'):
        plt.title(title)
        plt.scatter(X[y == 0, 0], X[y == 0, 1])
        plt.scatter(X[y == 1, 0], X[y == 1, 1])


torch.manual_seed(2)
model = Model(2, 4)
model.learn()

model.plot_decision_boundary(X, y)
model.plot_fit()
plt.show()

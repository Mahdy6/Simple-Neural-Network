import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns

class NeuralN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.b00 = nn.Parameter(torch.rand(1), requires_grad=False)

        self.w01 = nn.Parameter(torch.rand(1), requires_grad=False)
        self.b01 = nn.Parameter(torch.rand(1), requires_grad=False)

        self.w02 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.b02 = nn.Parameter(torch.rand(1), requires_grad=False)

        self.w11 = nn.Parameter(torch.rand(1), requires_grad=False)
        self.w12 = nn.Parameter(torch.rand(1), requires_grad=False)
        self.w13 = nn.Parameter(torch.rand(1), requires_grad=False)

        self.w20 = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, input):
        input_to_layer_01 = input * self.w00 + self.b00
        layer_01_output = torch.sigmoid(input_to_layer_01)
        layer_01_output = layer_01_output * self.w11

        input_to_layer_02 = input * self.w01 + self.b01
        layer_02_output = torch.sigmoid(input_to_layer_02)
        layer_02_output = layer_02_output * self.w12

        input_to_layer_03 = input * self.w02 + self.b02
        layer_03_output = torch.sigmoid(input_to_layer_03)
        layer_03_output = layer_03_output * self.w13

        output = layer_01_output + layer_02_output + layer_03_output

        output = torch.tanh(output)
        output = output * self.w20
        return output

my_model=NeuralN()

input_data = torch.rand(10, 3)  # Random input of size (10, 3)
predictions_before_training = my_model(input_data).detach().numpy()

sns.lineplot(data=predictions_before_training.flatten())
plt.title('Initial Predictions (Before Training)')
plt.show()

simpleNN = NeuralN()

X = torch.linspace(start=1, end=2.5, steps=40)
X

Y = my_model(X)
Y

newY=simpleNN(X)

newY

sns.set(style="whitegrid")

sns.lineplot(
    x=X,
    y=newY.detach(),
    color='red',
    linewidth=3
)


plt.xlabel('X')
plt.ylabel('Y')

optimizer = optim.SGD(simpleNN.parameters(), lr=0.01)
loss = nn.MSELoss()


for epoch in range(50):
    total_loss = 0

    for i in range(len(X)):
        input_i = X[i]
        actual_output_i = Y[i]

        pred_output_i = simpleNN(input_i)

        loss_value = loss(pred_output_i, actual_output_i)

        # Retain the graph for multiple backward passes
        loss_value.backward(retain_graph=True)

        total_loss += loss_value

    print('Epoch: ', epoch, ' | Total Loss: ', total_loss)
    optimizer.step()
    optimizer.zero_grad()


pred_y = simpleNN(X)
pred_y

sns.set(style="whitegrid")

sns.lineplot(
    x=X,
    y=pred_y.detach(),
    color='red',
    linewidth=3
)


plt.xlabel('X')
plt.ylabel('Y')
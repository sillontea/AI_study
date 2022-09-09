# 1.import packages

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# 2.Load Data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
)


batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape}, {y.dtype}")
    break

# 3.Create nn.Model
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(4,4), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=4, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1, padding=0)

        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # nn.functional.tanh is deprecated
        x = torch.tanh(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = torch.tanh(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = torch.tanh(self.conv3(x))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

model = Lenet()
print(model)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(next(model.parameters()).device)

# from torchsummary import summary
# summary(model, input_size=(1,28,28))


# 4.Define loss function and Optimizer
#loss function
loss_fn = nn.CrossEntropyLoss(reduction='sum')

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5.Train
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


# 6.Test
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 7.Model Train
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n"+"-"*20)
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Save Model
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# Load Model
"""
model = Lenet()
model.load_state_dict(torch.load("model.pth")
"""
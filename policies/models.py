import torch
import torch.nn as nn
import torch.nn.functional as F

# each player has 12 features
class CNN(nn.Module):
    def __init__(self, ac_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(23, 30, 12)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(30, 60, 6)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(60, 120, 3)
        self.pool2 = nn.MaxPool1d(3)
        self.fc1 = nn.Linear(120, 50)
        self.fc2 = nn.Linear(50, ac_dim)

    # input is flattened, unflatten it
    def forward(self, x):
        x = x.reshape(-1, 23, 12)
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
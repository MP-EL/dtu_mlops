from torch import nn
import torch.nn.functional as F
import torch

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(28,28,1)
        self.conv2 = nn.Conv2d(28,22,1)
        self.conv3 = nn.Conv2d(22,10,1)
        # self.fc1 = nn.Linear(784, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(280, 10)
        
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        # make sure input tensor is flattened
        # x = torch.flatten(x)
        # print("5", x.size())
        # print(x) 
        # print("6", x.size())
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        x = torch.flatten(x)
        # x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=0)
        return x
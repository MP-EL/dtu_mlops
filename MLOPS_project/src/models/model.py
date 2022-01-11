from torch import nn
import torch.nn.functional as F
import torch

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,4,kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(4,8,kernel_size=3, stride=1,padding=1)
        self.conv3 = nn.Conv2d(8,4,kernel_size=3, stride=1,padding=1)
        # self.fc1 = nn.Linear(784, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(3136, 10)
        
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        x = x.float().unsqueeze(dim=1)
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        # x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=0)
        return x
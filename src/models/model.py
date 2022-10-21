from torch import nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.tc1 = nn.Linear(input_size, 1024)
        self.tc2 = nn.Linear(1024, 1024)
        self.output = nn.Linear(1024, output_size)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout1(F.relu(self.tc1(x)))
        x = self.dropout2(F.relu(self.tc2(x)))
        x = self.output(x)

        return x
import torch
import torch.nn as nn

class myLinearModel(nn.Module):
    """
    Classic Linear model: Since matrices cannot be passed through Linear, we flatten
        - 10x10 to 100 to 2
    """
    def __init__(self, in_dimension=100, classes= 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dimension, classes)
        )

    def forward(self, x):
        return self.model(x)

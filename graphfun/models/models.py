import torch
import torch.nn as nn

#-----
#   Models
#-----

class myLinearModel(nn.Module):
    """
    Classic Linear model: with flatten, can we pass through tensor? 
        - 10x10 to 100 to 2
        - get_graphs now collects flatten vectors, so nn.Flatten() is unecessary
    """
    def __init__(self, in_dimension=100, classes= 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dimension, classes)
        )

    def forward(self, x):
        return self.model(x)

class myMLP(nn.Module):
    """
    MLP with one hidden layer, and ReLu activation 
    """
    def __init__(self, in_dimension=100, intermediate=50, classes= 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dimension, intermediate),
            nn.ReLU(),
            nn.Linear(intermediate, classes)
        )
    def forward(self, x):
        return self.model(x)

class myMLP2(nn.Module):
    """
    MLP with two hidden layer, and ReLu activation 
    """
    def __init__(self, in_dimension=100, intermediate1=50, intermediate2=25, classes= 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dimension, intermediate1),
            nn.ReLU(),
            nn.Linear(intermediate1, intermediate2),
            nn.ReLU(),
            nn.Linear(intermediate2, classes)
        )
    def forward(self, x):
        return self.model(x)

class myMLP3(nn.Module):
    """
    MLP with three hidden layers, and ReLu activation 
    """
    def __init__(self, in_dimension=100, intermediate1=75, intermediate2=50, intermediate3= 25, classes= 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dimension, intermediate1),
            nn.ReLU(),
            nn.Linear(intermediate1, intermediate2),
            nn.ReLU(),
            nn.Linear(intermediate3, intermediate2),
            nn.ReLU(),
            nn.Linear(intermediate3, classes)
        )
    def forward(self, x):
        return self.model(x)
    
class myAttention(nn.Module):
    """
    Actual Attention this time
    """
    def __init__(self, in_dimension=10):
        super().__init__()

        self.key 

class myLinearWithAttBad(nn.Module):
    """
    Uses attention as a learned weighted average over rows.
    self.attention_score is taking is treating the adjeacency matrix as a set of 10 vectors in \(\R^{10}\)
    in `def forward(self, x)`
    **Not attention**
        - each vector gets assigned a scalar, so we get a vector in \(\R^{10}\)
        - softmax normalizes across the 10 scores, turning them into a probability distribution
        - with `context = (weights * x).sum(dim=1)` each vector gets scaled by its scalar weight
            - then we sum collapsing the the set of ten vector into 1 vector in \(\R^{10}\)
        - The last linear layer produces a vector in \(\R^{2}\)
    """
    def __init__(self, in_dimension=10, classes=2):
        super().__init__()
        # Learns a score for each row
        self.attention_score = nn.Linear(in_dimension, 1)
        self.classifier = nn.Linear(in_dimension, classes)

    def forward(self, x):
        # x: (batch, 10, 10)
        scores = self.attention_score(x)          # (batch, 10, 1)
        weights = torch.softmax(scores, dim=1)    # (batch, 10, 1) — sums to 1 over rows
        context = (weights * x).sum(dim=1)        # (batch, 10) — weighted sum of rows
        return self.classifier(context)           # (batch, 2)


    


# ------------------------
# Model Registry (Factory)
# ------------------------

MODEL_REGISTRY = {
    "linear": myLinearModel,
    "MLP": myMLP,
    "MLP2": myMLP2,
    "MLP3": myMLP3,
    "linear_w_att_bad": myLinearWithAttBad
}

def get_model(name: str, **kwargs) -> nn.Module:
    """
    Factory function to instantiate models.

    Args:
        name (str): model name
        **kwargs: passed to model constructor

    Returns:
        nn.Module
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[name](**kwargs)


# ------------------------
# Utility (optional)
# ------------------------

def list_models():
    return list(MODEL_REGISTRY.keys())
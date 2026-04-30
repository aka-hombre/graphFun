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
    
class myLinearWithAtt(nn.Module):
    """
    0_0     
    """
    def __init__(self, in_dimension=10, classes= 2):
        super().__init__()
        self.model = nn.Linear(in_dimension, classes)

        def forward(self, x):
            return self.model(self.to_vec(x))

    def to_vec(X):
        return torch.split(X, 10, dim=0)    #   dim=0 corresponds to splitting by rows

    


# ------------------------
# Model Registry (Factory)
# ------------------------

MODEL_REGISTRY = {
    "linear": myLinearModel,
    "MLP": myMLP,
    "linear_w_att": myLinearWithAtt
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
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



# ------------------------
# Model Registry (Factory)
# ------------------------

MODEL_REGISTRY = {
    "linear": myLinearModel
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
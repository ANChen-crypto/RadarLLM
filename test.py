import torch
from modeling import MaskedAutoencoder


x = torch.randn(1, 60, 40, 40)


model = MaskedAutoencoder()
results = model(x)
print()
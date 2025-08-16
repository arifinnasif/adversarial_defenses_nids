import torch
import torch.nn as nn
# from config import output_dim

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, output_dim),
            # nn.Softmax(dim=1) # necessary as foolbox expects the output to be multiclass probabilities
        )

    def forward(self, x):
        # print("Input shape:", x.shape)
        # print("from MLP forward method")
        # print(x[0])
        return self.layers(x)

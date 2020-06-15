import torch


class PrintTensors(torch.nn.Module):

    def __init__(self, name, verbose=False):
        super().__init__()
        self.name = name
        self.verbose = verbose

    def forward(self, X):
        if self.verbose:
            print(f"Name: {self.name}, Size: {X.size()}")
        return X





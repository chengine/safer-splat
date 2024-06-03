import torch

class SingleIntegrator():
    def __init__(self, device, ndim=3):
        self.ndim = ndim
        self.device = device

    def system(self, x, u=None):

        # Defines the f function
        f = torch.zeros(self.ndim).to(self.device)
        g = torch.eye(self.ndim).to(self.device)
        return f, g

    
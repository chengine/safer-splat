import torch

class SingleIntegrator():
    def __init__(self, device, ndim=3):
        self.ndim = ndim
        self.device = device
        self.rel_deg = 1

    def system(self, x, u=None):

        # Defines the f function
        f = torch.zeros(self.ndim).to(self.device)
        g = torch.eye(self.ndim).to(self.device)
        return f, g

class DoubleIntegrator():
    def __init__(self, device, ndim=3):
        self.ndim = ndim
        self.device = device
        self.rel_deg = 2

    def system(self, x, u=None):
        """
        Defines the f and g functions for the double integrator system.
        x: state vector (position and velocity)
        u: control input (acceleration)
        """
        # Split state vector x into position and velocity
        pos = x[:self.ndim]
        vel = x[self.ndim:]

        # f function (dynamics without control input)
        f_pos = vel
        f_vel = torch.zeros(self.ndim).to(self.device)
        f = torch.cat((f_pos, f_vel))

        # g function (control input influence)
        g = torch.zeros(2*self.ndim, self.ndim).to(self.device)
        g[self.ndim:, :] = torch.eye(self.ndim).to(self.device)

        # A matrix (df/dx)
        A = torch.zeros(2*self.ndim, 2*self.ndim).to(self.device)
        A[:self.ndim, self.ndim:] = torch.eye(self.ndim).to(self.device)

        return f, g, A


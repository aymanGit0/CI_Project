import numpy as np
from lib.Layers import Layer


class Tanh(Layer):
    """Hyperbolic Tangent activation function."""
    def forward(self, x):
        # Save output for use in the backward pass
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        # Local gradient: 1 - tanh^2(x) = 1 - output^2
        # Tanh(x) is stored in self.output
        local_grad = 1 - self.output**2
        return grad_output * local_grad
    
class Sigmoid(Layer):
    """Sigmoid activation function."""
    def forward(self, x):
        # Save output for use in the backward pass
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        # Local gradient: output * (1 - output)
        # Sigmoid(x) is stored in self.output
        local_grad = self.output * (1 - self.output)
        return grad_output * local_grad
    

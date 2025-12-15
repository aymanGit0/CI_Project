import numpy as np
from Layers import Layer


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
    
class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) activation.
    f(x) = max(0, x)
    """
    def forward(self, x):
        # Save input for backward pass
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        # Gradient is 1 where x > 0, else 0
        return grad_output * (self.x > 0)
    
class Softmax(Layer):
    """
    Softmax activation function (usually for output layer classification).
    Numerically stable implementation.
    """
    def forward(self, x):
        # Shift x by subtracting max to prevent overflow (numerical stability)
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(x_shifted)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        # The gradient of Softmax is complex on its own (Jacobian matrix).
        # We implement the efficient vector-Jacobian product here.
        # dL/dx = y * (dL/dy - sum(dL/dy * y))
        
        # 1. Calculate dot product of (grad_output . output) per sample
        # shape: (batch_size, 1)
        sum_grad = np.sum(grad_output * self.output, axis=1, keepdims=True)
        
        # 2. Combine
        grad_input = self.output * (grad_output - sum_grad)
        return grad_input

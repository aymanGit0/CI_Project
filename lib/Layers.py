"""
Layer base classes and Dense implementation for the mini NN library.

Contains:
- Layer (abstract base)
- Dense (fully-connected) with forward/backward and Xavier init
"""

import numpy as np


class Layer:
    """Abstract base class for layers."""
    def forward(self, x):
        """Compute forward pass. Save anything needed for backward."""
        raise NotImplementedError

    def backward(self, grad_output):
        """Compute backward pass. Return gradient w.r.t. layer input."""
        raise NotImplementedError


class Dense(Layer):
    """
    Fully connected layer.

    Attributes:
        W: weight matrix of shape (in_features, out_features)
        b: bias row vector of shape (1, out_features)
        x: saved input from forward pass
        dW: gradient of loss w.r.t W (computed in backward)
        db: gradient of loss w.r.t b (computed in backward)
    """

    def __init__(self, in_features, out_features, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Xavier / Glorot uniform initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, size=(in_features, out_features))
        self.b = np.zeros((1, out_features))

        # Grad placeholders (will be created after backward)
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Forward pass: save input and return X @ W + b
        x shape: (batch_size, in_features)
        returns: (batch_size, out_features)
        """
        assert x.ndim == 2, "Input to Dense.forward must be 2D (batch_size, in_features)"
        self.x = x  # save for backward
        return x @ self.W + self.b  # matrix multiply + bias broadcast

    def backward(self, grad_output):
        """
        Backward pass:
        - grad_output: dL/dZ where Z = X @ W + b (shape: batch_size x out_features)
        Returns:
        - grad_input: dL/dX (shape: batch_size x in_features)
        Also sets:
        - self.dW: dL/dW (shape: in_features x out_features)
        - self.db: dL/db (shape: 1 x out_features)
        """
        # dW = X^T @ grad_output
        self.dW = self.x.T @ grad_output  # shape (in_features, out_features)

        # db is sum over batch rows
        self.db = np.sum(grad_output, axis=0, keepdims=True)  # shape (1, out_features)

        # grad_input = grad_output @ W^T
        grad_input = grad_output @ self.W.T  # shape (batch_size, in_features)
        return grad_input

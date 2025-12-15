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

class Flatten(Layer):
    """
    Reshapes a 3D input (batch_size, H, W) to a 2D output (batch_size, H*W).
    Needed for image data (e.g., 28x28) to be used by Dense layers.
    """
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        """
        Flattens the input image/tensor into a vector.
        x shape: (batch_size, H, W) -> (N, H*W)
        """
        # Assuming input shape is (batch_size, H, W)
        self.input_shape = x.shape
        # -1 automatically calculates the total number of elements in the row (H*W)
        return x.reshape(x.shape[0], -1) 

    def backward(self, grad_output):
        """
        Reshapes the incoming gradient back to the original input shape.
        grad_output shape: (N, H*W) -> (batch_size, H, W)
        """
        # Reshape the gradient back to the original shape
        return grad_output.reshape(self.input_shape)


class Reshape(Layer):
    """
    Reshapes the input tensor to a target shape.
    Used for the Decoder output to return the image to its original (28, 28) shape.
    """
    def __init__(self, target_shape):
        """
        target_shape is the desired shape *after* the batch dimension.
        e.g., (28, 28) for an MNIST image.
        """
        self.target_shape = target_shape
        self.input_shape = None # Will store the batch-inclusive input shape

    def forward(self, x):
        """
        x shape: (batch_size, features)
        returns: (batch_size, target_shape[0], target_shape[1], ...)
        """
        self.input_shape = x.shape
        # Prepend batch dimension (x.shape[0]) to the target shape
        return x.reshape(x.shape[0], *self.target_shape)

    def backward(self, grad_output):
        """
        Flattens the incoming gradient back to the 2D shape expected by the preceding layer.
        grad_output shape: (batch_size, H, W) -> (N, H*W)
        """
        # Reshape the gradient back to the 2D shape it had before this layer
        return grad_output.reshape(self.input_shape)
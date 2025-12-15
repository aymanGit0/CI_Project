import numpy as np
import random


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    - Randomly selects ONE sample from data
    - Performs forward, backward
    - Updates parameters
    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, model, X, Y, loss_fn):
        """
        Perform ONE stochastic update step.

        Args:
            model: neural network model (has forward, backward, layers)
            X (np.ndarray): input data (N, ...)
            Y (np.ndarray): target data (N, ...)
            loss_fn: loss function object
        """

        # 1. Pick ONE random sample
        i = random.randint(0, len(X) - 1)
        x = X[i:i+1]  # keep batch dimension
        y = Y[i:i+1]

        # 2. Zero gradients
        for layer in model.layers:
            if hasattr(layer, "dW"):
                layer.dW = np.zeros_like(layer.W)
            if hasattr(layer, "db"):
                layer.db = np.zeros_like(layer.b)

        # 3. Forward pass
        y_pred = model.forward(x)

        # 4. Compute loss
        loss = loss_fn.loss(y, y_pred)

        # 5. Backward pass
        dL = loss_fn.gradient()
        model.backward(dL)

        # 6. Parameter update
        eta = self.learning_rate
        for layer in model.layers:
            if hasattr(layer, 'W'):
                layer.W -= eta * layer.dW
            if hasattr(layer, 'b'):
                layer.b -= eta * layer.db

        return loss

import numpy as np
import random


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    - Randomly selects ONE sample from data
    - Performs forward, backward
    - Updates parameters
    """

    def __init__(self, learning_rate,batch_size):
        self.learning_rate = learning_rate
        self.batch_size=batch_size

    def step(self, model, X, Y, loss_fn,batch_size):
        """
        Perform ONE mini-batch SGD update.
        """
        self.batch_size=batch_size
        N = len(X)

        # 1. Sample a mini-batch
        indices = np.random.choice(N, self.batch_size, replace=False)
        x_batch = X[indices]
        y_batch = Y[indices]    

        # 2. Forward pass
        y_pred = model.forward(x_batch)

        # 3. Compute loss
        loss = loss_fn.loss(y_batch, y_pred)

        # 4. Backward pass
        grad = loss_fn.gradient()
        model.backward(grad)

        # 5. Parameter update
        for layer in model.layers:
            if hasattr(layer, "W"):
                layer.W -= self.learning_rate * layer.dW
            if hasattr(layer, "b"):
                layer.b -= self.learning_rate * layer.db

        return loss

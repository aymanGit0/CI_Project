import numpy as np
from Layers import Layer

class MeanSquaredError:
    """
    Mean Squared Error (MSE) loss function with 1/(2N) scaling.
    
    L = 1/(2N) * sum((Y_true - Y_pred)^2)
    """

    def forward(self, Y_pred, Y_true):
        """
        Calculates the MSE loss.
        
        Args:
            Y_pred (np.ndarray): The network's prediction.
            Y_true (np.ndarray): The true target values.
            
        Returns:
            float: The scalar loss value.
        """
        self.diff = Y_pred - Y_true # (Y_pred - Y_true)
        self.N = Y_pred.shape[0]    # Batch size (N)
        
        # Loss = 1/(2N) * sum(diff^2)
        loss = np.sum(self.diff**2) / (2 * self.N)
        return loss

    def backward(self):
        """
        Calculates the initial gradient of the loss with respect to the prediction.
        
        dL / dY_pred = 1/N * (Y_pred - Y_true)
        
        Returns:
            np.ndarray: The gradient dL/dY_pred, with shape identical to Y_pred.
        """
        # (Y_pred - Y_true) is stored as self.diff
        # dL/dY_pred = self.diff / self.N
        grad_output = self.diff / self.N
        return grad_output
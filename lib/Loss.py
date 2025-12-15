import numpy as np

class MeanSquaredError:
    """
    Mean Squared Error (MSE) loss function.
    Robust to 2D (N, 784) and 3D (N, 28, 28) inputs.
    """
    def __init__(self):
        self.diff = None 
        self.N = None 
        self.F = None    
        self.Loss = None
        self.grad = None

    def loss(self, Y_true, Y_pred):
        """
        Calculates the MSE loss and STORES self.diff for gradient calculation.
        """
        # 1. Store the difference (Crucial for gradient step)
        self.diff = Y_pred - Y_true
        
        # 2. Get dimensions robustly
        self.N = Y_pred.shape[0]
        # Calculate features (F) as product of remaining dimensions
        self.F = np.prod(Y_pred.shape[1:])

        # 3. Compute scalar loss
        squared_diff = self.diff ** 2
        total_elements = self.N * self.F 
        
        self.Loss = np.sum(squared_diff) / (2 * total_elements)
        return self.Loss

    def gradient(self):
        """
        Calculates dL/dY_pred using the stored self.diff.
        """
        if self.diff is None:
            raise RuntimeError("Must call .loss() before .gradient() to store the error term.")

        total_elements = self.N * self.F
        
        # Gradient = (Y_pred - Y_true) / (N * F)
        self.grad = self.diff / total_elements 
        return self.grad
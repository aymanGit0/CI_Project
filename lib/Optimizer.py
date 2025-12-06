import numpy as np
from lib.Layers import Layer


class GD:
    """
    Stochastic Gradient Descent optimizer.

    Updates parameters (W, b) of layers using their gradients (dW, db)
    and a defined learning rate (eta).
    """
    
    def __init__(self, learning_rate):
        """
        Initialize the GD optimizer.
        
        Args:
            learning_rate (float): The step size (eta) for parameter updates.
        """
        self.learning_rate = learning_rate

    def step(self, layers):
        """
        Updates the parameters (W, b) for all trainable layers in the network.
        
        The update rule is: W_new = W_old - eta * (dL/dW)
        
        Args:
            layers (list): A list of Layer objects in the network.
        """
        eta = self.learning_rate
        
        # Iterate over all layers provided by the Network/Sequential model
        for layer in layers:
            
            # 1. Check for trainable weights (W and dW)
            # This handles the Dense layer
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                # W_new = W_old - eta * dW
                layer.W -= eta * layer.dW
                
            # 2. Check for trainable biases (b and db)
            # This handles the Dense layer
            if hasattr(layer, 'b') and hasattr(layer, 'db'):
                # b_new = b_old - eta * db
                layer.b -= eta * layer.db



import numpy as np

class Network:
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None

    def add(self, layer):
        """Adds a layer to the network."""
        self.layers.append(layer)

    def compile(self, loss_instance, optimizer_instance):
        """Sets the loss function and optimizer."""
        self.loss_fn = loss_instance
        self.optimizer = optimizer_instance
        print("Network compiled successfully.") 
        
    def forward(self, X):
        """Performs the forward pass through all layers."""
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad_output):
        """Performs the backward pass."""
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    # --- UPDATED TRAIN METHOD ---
    def train(self, x_train, y_train, iterations, batch_size):
        if self.loss_fn is None or self.optimizer is None:
            raise RuntimeError("Model not compiled. Call .compile() first.")
        
        # Update the optimizer's batch size with the one passed here
        if hasattr(self.optimizer, 'batch_size'):
            self.optimizer.batch_size = batch_size

        print_frequency = max(1, iterations // 10)

        for i in range(iterations):
            # We use the optimizer.step() to handle:
            # 1. Mini-batch sampling
            # 2. Forward pass
            # 3. Backward pass
            # 4. Weight updates
            loss = self.optimizer.step(self, x_train, y_train, self.loss_fn, batch_size)

            # Logging
            if (i + 1) % print_frequency == 0 or (i + 1) == iterations: 
                print(f"Iteration {i+1}/{iterations}, Loss: {loss:.8f}")

        print(f"Iteration {iterations}/{iterations}, Final Loss: {loss:.8f}")
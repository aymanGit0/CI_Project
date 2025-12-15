class Network:
    def __init__(self):
        self.layers = []
        self.loss_fn = None  # Stores the MeanSquaredError instance
        self.optimizer = None

    def add(self, layer):
        """Adds a layer or activation function to the network."""
        self.layers.append(layer)

    def compile(self, loss_instance, optimizer_instance):
        """Initializes the loss function instance and the optimizer instance."""
        self.loss_fn = loss_instance
        self.optimizer = optimizer_instance
        print("Network compiled successfully.") 
        
    def forward(self, X):
        """Performs the forward pass through all layers."""
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad_output):
        """
        Performs the backward pass (backpropagation) through all layers,
        starting with the initial gradient from the loss function.
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def train(self, x_train, y_train, iterations):
        if self.loss_fn is None or self.optimizer is None:
            raise RuntimeError("Model not compiled. Call .compile() first.")

        for i in range(iterations):
            # 1. Forward Pass
            output = self.forward(x_train)
            
            # 2. Calculate Loss Value
            error = self.loss_fn.loss(y_train, output)
            
            # 3. Calculate Initial Gradient
            grad = self.loss_fn.gradient()
            
            # 4. Backward Pass (Backpropagation)
            self.backward(grad)
            
            # 5. Optimizer Step
            # Assuming your optimizer object (GD) has a method named 'step'
            self.optimizer.step(self, x_train, y_train, self.loss_fn)

            # Logging
            if (i + 1) % 1000 == 0:
                print(f"iteration {i+1}/{iterations}, Loss: {error:.8f}")

        # Print final loss after the last iteration
        # Recalculate output and loss one last time if desired, or just print the last recorded error.
        print(f"iteration {iterations}/{iterations}, Final Loss: {error:.8f}")

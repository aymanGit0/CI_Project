class Network:
    # FIX 1: Constructor must be __init__ (double underscore)
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_gradient = None  # Renamed for consistency
        self.optimizer = None   # Renamed for consistency

    def add(self, layer):
        # FIX 2: Fixed typo (layerS -> layers)
        self.layers.append(layer)

    def compile(self, loss_class, optimizer):
        self.loss = loss_class.loss
        self.loss_gradient = loss_class.gradient
        self.optimizer = optimizer # Stored as self.optimizer

    def forward(self, input_data):
        # FIX 3: The data must flow through the layers!
        # Input -> Layer 1 -> Output -> Layer 2 -> Output...
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, gradient):
        # FIX 4: Update the gradient at every step
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def train(self, x_train, y_train, epochs):
        # FIX 5: Loop range was named 'iterations', passed as 'epochs'
        for i in range(epochs):
            # 1. Forward Pass
            output = self.forward(x_train)
            
            # 2. Calculate Error (Scoreboard)
            error = self.loss(y_train, output)
            
            # 3. Backward Pass
            grad = self.loss_gradient(y_train, output)
            self.backward(grad)
            
            # 4. Optimizer Step
            # Now variable names match (self.optimizer)
            self.optimizer.step(self.layers)

            # Logging
            if (i + 1) % 1000 == 0:
                print(f"Epoch {i+1}/{epochs}, Loss: {error}")
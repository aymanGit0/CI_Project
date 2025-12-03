class Network:
    def __init__(self):
        self.layers = []
        self.loss = None  # holds the loss function
        self.loss_gradient = None  # holds the loss gradient function
        self.optimizer = None   

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss_class, optimizer):
        self.loss = loss_class.loss  # stores the loss function 
        self.loss_gradient = loss_class.gradient  # stores the loss gradient function
        self.optimizer = optimizer # Stored as self.optimizer

    def forward(self, input_data):
        # The data must flow through the layers!
        # Input -> Layer 1 -> Output -> Layer 2 -> Output...
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, gradient):
        # Update the gradient at every step
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def train(self, x_train, y_train, iterations):
        # Loop range was named 'iterations', passed as 'iterations'
        for i in range(iterations):
            # 1. Forward Pass
            output = self.forward(x_train)
            
            # 2. Calculate Error (Scoreboard)
            error = self.loss(self,y_train, output)
            
            # 3. Backward Pass
            grad = self.loss_gradient(self)
            self.backward(grad)
            
            # 4. Optimizer Step
            # Now variable names match (self.optimizer)
            self.optimizer.step(self.layers)

            # Logging
            if (i + 1) % 1000 == 0:
                print(f"iteration {i+1}/{iterations}, Loss: {error}")
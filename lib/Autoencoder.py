from lib.Layers import Dense, Flatten, Reshape
from lib.Activation import ReLU, Sigmoid, Tanh
from lib.Loss import MeanSquaredError
from lib.Optimizer import SGD
from lib.Network import Network

# --- UPDATED ARCHITECTURE CLASSES ---

class Encoder(Network):
    """
    Encodes (N, 28, 28) images into a compact latent vector (N, latent_dim).
    Architecture: Flatten -> Dense(784->256) -> ReLU -> Dense(256->Latent) -> ReLU
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.add(Flatten())                  # (N, 28, 28) -> (N, 784)
        self.add(Dense(784, 256))
        self.add(Tanh())
        self.add(Dense(256, latent_dim))
        self.add(Tanh())

class Decoder(Network):
    """
    Decodes latent vector (N, latent_dim) back to image (N, 28, 28).
    Architecture: Dense(Latent->256) -> ReLU -> Dense(256->784) -> Sigmoid -> Reshape
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.add(Dense(latent_dim, 256))
        self.add(Tanh())
        self.add(Dense(256, 784))
        self.add(ReLU())                  # Output pixels [0, 1]
        self.add(Reshape((28, 28)))          # (N, 784) -> (N, 28, 28)

class Autoencoder(Network):
    """
    Combines Encoder and Decoder.
    Flattens the layers list so the existing SGD optimizer works automatically.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Instantiate the two components
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
        # CRITICAL: Combine layers into self.layers so SGD and backprop work 
        # seamlessly with your existing Network/Optimizer logic.
        self.layers = self.encoder.layers + self.decoder.layers

    def forward(self, X):
        # We can implement forward using the sub-networks for clarity
        latent = self.encoder.forward(X)
        reconstruction = self.decoder.forward(latent)
        return reconstruction
        
    # backward() is handled by the parent Network class iterating over self.layer

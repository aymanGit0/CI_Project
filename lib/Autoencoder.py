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
    def __init__(self, input_shape,latent_dim, hidden_activation=ReLU, output_activation=Tanh):
        super().__init__()
        flat_size = input_shape[0] * input_shape[1]
        self.add(Flatten())  #convert the image from (N, 28, 28) to (N, 784)
        self.add(Dense(flat_size, 256))
        self.add(hidden_activation())
        self.add(Dense(256, latent_dim))
        self.add(output_activation())

class Decoder(Network):
    """
    Decodes latent vector (N, latent_dim) back to image (N, 28, 28).
    Architecture: Dense(Latent->256) -> ReLU -> Dense(256->784) -> Sigmoid -> Reshape
    """
    def __init__(self, input_shape,latent_dim, hidden_activation=ReLU, output_activation=Sigmoid):
        super().__init__()
        flat_size = input_shape[0] * input_shape[1]
        self.add(Dense(latent_dim, 256))
        self.add(hidden_activation())
        self.add(Dense(256, flat_size))
        self.add(output_activation())                  # Output pixels [0, 1]
        self.add(Reshape((input_shape[0], input_shape[1])))          # (N, 784) -> (N, 28, 28)

class Autoencoder(Network):
    """
    Combines Encoder and Decoder.
    Flattens the layers list so the existing SGD optimizer works automatically.
    """
    def __init__(self, image_shape,latent_dim, Encoder_hidden_act=ReLU, Encoder_out_act=Tanh, Decoder_hidden_act=ReLU, Decoder_out_act=Sigmoid):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Instantiate the two components
        self.encoder = Encoder(image_shape,latent_dim,Encoder_hidden_act, Encoder_out_act)
        self.decoder = Decoder(image_shape,latent_dim, Decoder_hidden_act, Decoder_out_act)
        
        # CRITICAL: Combine layers into self.layers so SGD and backprop work 
        # seamlessly with your existing Network/Optimizer logic.
        self.layers = self.encoder.layers + self.decoder.layers

    def forward(self, X): # this name can overwrite the parent class forward
        # We can implement forward using the sub-networks for clarity
        latent = self.encoder.forward(X)
        reconstruction = self.decoder.forward(latent)
        return reconstruction
        
    # backward() is handled by the parent Network class iterating over self.layer

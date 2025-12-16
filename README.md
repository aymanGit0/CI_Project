# Minimal NumPy Neural Network & Autoencoder Framework

This repository implements a small but complete deep learning framework using only NumPy, plus a set of experiments that demonstrate:

- How forward and backward passes are implemented from scratch  
- How to verify gradients numerically  
- How to solve the XOR problem with a multilayer perceptron  
- How to build, train, and evaluate an MNIST autoencoder, and use its latent space for SVM classification  

The implementation is educational, focusing on clarity and step‑by‑step experimentation rather than performance.

## 1. Repository Structure

```text
lib/
  Activation.py
  Autoencoder.py
  Layers.py
  Loss.py
  Network.py
  Optimizer.py
  __init__.py
notebook/
  project_demo.ipynb
report/
  (project report files)
README.md
.gitignore
```

- All core framework code lives in `lib/`.  
- `notebook/project_demo.ipynb` contains the full walkthrough: gradient checking, XOR training, MNIST autoencoder, Keras comparisons, and SVM classification using latent features.  
- The old `Test_Folder` used for validation has been removed in the current version.

## 2. Core Framework (lib/)

### 2.1 Network

`lib/Network.py` defines the `Network` class, a simple container similar to Keras’ Sequential API.

- `add(layer)`: append a layer  
- `compile(loss_instance, optimizer_instance)`: set loss and optimizer  
- `forward(X)`: pass data through all layers in order  
- `backward(grad_output)`: propagate gradients backward through all layers  
- `train(x_train, y_train, iterations, batch_size)`: training loop that repeatedly calls the optimizer’s `step` for mini‑batch updates and prints loss periodically

### 2.2 Layers

`lib/Layers.py` contains these layer types:

- `Layer`: abstract base with `forward` and `backward`  
- `Dense(in_features, out_features, seed=None)`  
  - Xavier/Glorot uniform initialization  
  - `forward`: computes `X @ W + b` with shape checks  
  - `backward`: computes `dW`, `db`, and gradient with respect to input  
- `Flatten`  
  - `forward`: reshapes `(batch_size, H, W)` into `(batch_size, H*W)`  
  - `backward`: reshapes gradients back to `(batch_size, H, W)`  
- `Reshape(target_shape)`  
  - `forward`: reshapes flat vectors into `(batch_size, *target_shape)` (e.g., `(N, 28, 28)`)  
  - `backward`: flattens gradients back to the 2D shape expected by the previous layer

### 2.3 Activations

`lib/Activation.py` defines activation layers with both forward and backward behavior:

- `Tanh`: uses `np.tanh`, gradient `1 - output**2`  
- `Sigmoid`: standard sigmoid, gradient `output * (1 - output)`  
- `ReLU`: `max(0, x)` with gradient 1 where `x > 0`, else 0  
- `Softmax`: numerically stable implementation (shift by max) with a vectorized Jacobian–vector product in `backward`

All activation classes inherit from `Layer` and plug into the same training pipeline.

### 2.4 Loss

`lib/Loss.py` provides `MeanSquaredError`, which supports both 2D and 3D tensors:

- `loss(Y_true, Y_pred)`  
  - Stores `diff = Y_pred - Y_true`  
  - Computes batch size `N` and feature count `F` (product of remaining dimensions)  
  - Returns scalar loss  

- `gradient()`  
  - Uses stored `diff` to compute `(Y_pred - Y_true) / (N * F)`

### 2.5 Optimizer

`lib/Optimizer.py` implements mini‑batch stochastic gradient descent:

- `SGD(learning_rate, batch_size)`  
- `step(model, X, Y, loss_fn, batch_size)`  
  - Samples a random mini‑batch  
  - Runs forward pass  
  - Computes loss via the loss function  
  - Gets the gradient from the loss  
  - Runs backward pass  
  - Updates any layer with `W` and/or `b`  

This keeps the parameter‑update logic in the optimizer while `Network.train` manages the outer loop.

### 2.6 Autoencoder Components

`lib/Autoencoder.py` defines reusable components for building an image autoencoder:

- `Encoder(image_shape, latent_dim, hidden_activation=ReLU, output_activation=Tanh)`  
  - Architecture: `Flatten → Dense(784→256) → hidden_activation → Dense(256→latent_dim) → output_activation`  
- `Decoder(image_shape, latent_dim, hidden_activation=ReLU, output_activation=Sigmoid)`  
  - Architecture: `Dense(latent_dim→256) → hidden_activation → Dense(256→784) → output_activation → Reshape(image_shape)`  
- `Autoencoder(image_shape, latent_dim, Encoder_hidden_act=ReLU, Encoder_out_act=Tanh, Decoder_hidden_act=ReLU, Decoder_out_act=Sigmoid)`  
  - Creates an `encoder` and `decoder`  
  - Concatenates their `layers` into `self.layers` so existing `Network` / `SGD` logic works unchanged  
  - Overrides `forward(X)` to explicitly compute `latent = encoder.forward(X)` and `reconstruction = decoder.forward(latent)`

## 3. Notebook Overview (project_demo.ipynb)

The notebook acts as a guided tour of the framework and contains three main sections.

### 3.1 Gradient Checking on a Minimal Network

- Builds a tiny fully connected network (e.g., 2–3–1 architecture)  
- Runs a forward pass on a small XOR‑style dataset  
- Implements numerical gradient checking using finite differences  
- Compares analytical gradients from `backward()` with numerical gradients for each weight and bias  
- Prints relative differences and reports success when the error is below a tight threshold

This validates the correctness of the backpropagation implementation.

### 3.2 Solving the XOR Problem

- Defines the classic XOR dataset: four input points with labels 0 or 1  
- Model: `Dense(2→4) → Tanh → Dense(4→1) → Sigmoid`  
- Loss: `MeanSquaredError`  
- Optimizer: `SGD` with a relatively high learning rate and batch size 1  
- Trains for 20,000 iterations, showing loss decreasing to a very small value  
- After training, computes predictions for all four input patterns, rounds them to {0,1}, prints them alongside true labels, and reports 100% accuracy

The notebook also includes a small Keras implementation of the same XOR model for comparison, with training time and loss printed.

### 3.3 MNIST Autoencoder & Feature Extraction

#### 3.3.1 Data Loading and Checks

- Uses `tensorflow.keras.datasets.mnist` to load 60,000 training and 10,000 test images (28×28 grayscale) and their labels  
- Converts to `float32` and normalizes pixel values to `[0, 1]`  
- Performs a train–test uniqueness check:  
  - Hashes each training image into a set  
  - Checks every test image to confirm that there are no duplicates between train and test sets  
  - Prints whether duplicates were found or the sets are fully disjoint

#### 3.3.2 Training the Custom NumPy Autoencoder

- Instantiates `Autoencoder(image_shape=(28, 28), latent_dim=64)` with ReLU/Tanh/Sigmoid activations  
- Compiles with `MeanSquaredError` loss and `SGD` optimizer (learning rate around 0.2, batch size 128)  
- Uses a manual training loop over many epochs:  
  - Each epoch iterates over mini‑batches, calling `optimizer.step` to update weights  
  - Averages loss per epoch and appends to `loss_history`  
  - Prints epoch number, average loss, and total training time  
- Plots MSE loss vs epoch, showing a steady decrease to a small reconstruction error

#### 3.3.3 Keras Baseline Autoencoder

- Builds a Keras autoencoder with an encoder and decoder using `Dense` layers and activations similar to the custom model  
- Flattens 28×28 images to 784‑dimensional vectors as input  
- Trains the Keras model with MSE loss and SGD (or similar setup)  
- Logs training and validation loss per epoch  
- Measures training duration and final reconstruction loss  
- Prints both Keras and custom autoencoder training times and final losses for comparison

#### 3.3.4 Reconstruction Visualization

- Selects a small batch of test images (e.g., 10 digits)  
- Uses the Keras model to generate reconstructions  
- Uses the custom NumPy autoencoder’s `forward` method to reconstruct the same images  
- Plots original and reconstructed images using Matplotlib:  
  - 2 rows of subplots: originals on top, reconstructions on the bottom  
  - Axes are hidden and grayscale colormap is used for clarity

#### 3.3.5 Latent Features + SVM Classification

- Passes the entire training and test sets through the custom encoder:  
  - `X_train_latent = autoencoder.encoder.forward(x_train)`  
  - `X_test_latent  = autoencoder.encoder.forward(x_test)`  
- Each image is compressed from 784 dimensions to a `latent_dim` (e.g., 64) feature vector  
- Trains an SVM classifier (`sklearn.svm.SVC` with RBF kernel) on the latent features and corresponding digit labels  
- Evaluates on the test latent features:  
  - Reports final test accuracy (around high 97–98%)  
  - Prints a full classification report with precision, recall, and F1 score per class  
  - Plots a confusion matrix heatmap using Seaborn with clear labels and a title that includes the final accuracy

This section demonstrates that the learned latent representations are highly informative for downstream classification tasks.

## 4. Installation

Minimal requirement:

```bash
pip install numpy
```

To run the notebook fully (autoencoder, visualizations, SVM, Keras baselines), install:

```bash
pip install matplotlib seaborn scikit-learn tensorflow
```

## 5. Quick Start

### Run the Notebook

1. Open `notebook/project_demo.ipynb` in Jupyter, JupyterLab, or VS Code.  
2. Run all cells in order to:  
   - Verify gradients on a simple network  
   - Train and evaluate the XOR model  
   - Train the MNIST autoencoder (NumPy and Keras)  
   - Visualize reconstructions  
   - Train and evaluate an SVM on latent features

### Use the Library in Your Own Scripts

Example minimal usage:

```python
from lib.Network import Network
from lib.Layers import Dense
from lib.Activation import ReLU, Sigmoid
from lib.Loss import MeanSquaredError
from lib.Optimizer import SGD

model = Network()
model.add(Dense(10, 32))
model.add(ReLU())
model.add(Dense(32, 1))
model.add(Sigmoid())

model.compile(loss_instance=MeanSquaredError(),
              optimizer_instance=SGD(learning_rate=0.1, batch_size=16))

model.train(X_train, Y_train, iterations=1000, batch_size=16)
```

## 6. Summary

This project provides:

- A modular NumPy neural‑network framework (layers, activations, loss, optimizer, autoencoder classes)  
- Verified backpropagation via numerical gradient checking  
- A working XOR example with 100% accuracy and a Keras baseline for comparison  
- A custom MNIST autoencoder with low reconstruction error  
- A Keras autoencoder baseline for speed and loss comparison  
- Demonstration that autoencoder latent features support a strong SVM classifier on MNIST with high test accuracy

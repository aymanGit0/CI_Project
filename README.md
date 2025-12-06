# Minimal Neural Network Framework (NumPy-based)

This repository contains a simple, modular, and purely NumPy-based implementation of a basic deep learning framework. It is designed for educational purposes to understand the fundamentals of Neural Networks, including the forward pass, backpropagation, and gradient-based optimization, without relying on high-level libraries like TensorFlow or PyTorch.

Features

This framework allows users to define and train fully connected neural networks using a simple, Keras-like API.

Core Components

Component

Description

Network.py

The main container class for the model. Handles stacking layers, compiling the model with loss/optimizer, and running the train, forward, and backward loops.

layers.py

Contains trainable layers, such as the Dense (fully connected) layer. Implements forward and backward methods for weight and bias updates.

Activation.py

Contains activation functions (e.g., Tanh, Sigmoid). Implements both the forward computation and the local gradient calculation.

Loss.py

Contains the loss function (e.g., MeanSquaredError). Implements the main loss calculation and the initial gradient for backpropagation.

Optimizer.py

Contains optimization algorithms (e.g., SGD - Stochastic Gradient Descent). Implements the step method to apply weight and bias updates based on calculated gradients.

Key Functionality

Modular Design: Easily swap out different optimizers, layers, or loss functions.

Stateful Loss: The Loss function (e.g., MSE) maintains state to correctly calculate the initial gradient ($\frac{dL}{dY_{pred}}$).

Gradient Checking: The framework includes tools (numerical_gradient) to verify the correctness of the analytical backpropagation against numerical approximations.

Installation and Setup

Since this framework is built entirely on standard Python libraries, setup is minimal.

Prerequisites

You only need NumPy:

pip install numpy



Repository Structure

.
├── lib/
│   ├── Activation.py
│   ├── layers.py
│   ├── Loss.py
│   ├── Network.py
│   └── Optimizer.py
├── notebook/
│   └── project_demo.ipynb  # Example usage and gradient checking
├── Test_Folder/
│   └── validation.py       # Validation and testing utilities
└── report/
    └── CI_Project_Milestone_1.pdf # Project documentation



Usage Example (XOR Problem)

The primary example demonstrating model instantiation, training, and testing (including the full XOR problem solution and gradient check) is located in the Jupyter notebook:

notebook/project_demo.ipynb

Quick API Preview

The setup follows this simple sequence:

from lib.Network import Network
... imports for Dense, Tanh, Sigmoid, MSE, SGD

1. Instantiate the model
model = Network()
model.add(Dense(2, 3)) 
model.add(Tanh())
model.add(Dense(3, 1))
model.add(Sigmoid())

2. Compile
model.compile(loss_instance=MeanSquaredError(), 
              optimizer_instance=SGD(learning_rate=0.1))

3. Train
model.train(X_data, Y_true, EPOCHS)


Gradient Verification

To ensure the backpropagation implementation is correct, the framework is designed to pass the Gradient Check test, which compares the analytically calculated gradients (dL/dW from backward()) against the numerically calculated gradients (using the Finite Difference Method).

The utility function numerical_gradient() is used within the project_demo.ipynb notebook to confirm numerical stability and correctness of all trainable parameters.

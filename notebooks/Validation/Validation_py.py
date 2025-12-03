import numpy as np
import sys
import os 

sys.path.append(os.path.abspath('../..'))

from lib.Network import Network 
from lib.Layers import Dense
from lib.Activation import Tanh, Sigmoid
from lib.Loss import MeanSquaredError as MSE
from lib.Optimizer import SGD


#Creating the deta set:
X_data = np.array([[0,0],[0,1],[1,0],[1,1]])
Label = np.array([[0],[1],[1],[0]])
#---------------------

XOR_Model = Network()

# Add Layers (Architecture: 2 input nodes -> 4 Hidden nodes  -> 1 Output node)
XOR_Model.add(Dense(2, 4))
XOR_Model.add(Tanh()) # Activation function for hidden layer
XOR_Model.add(Dense(4, 1))
XOR_Model.add(Sigmoid())   # Activation function for output (forces 0-1)
#------------------
# Create optimizer with learning rate 0.1 & Compile with MSE loss and our optimizer
opt = SGD(learning_rate=0.1)

XOR_Model.compile(MSE, opt)
#-------------------
# Train for 10,000 epochs
XOR_Model.train(X_data, Label,10000)

# Check if it actually learned
print("\n--- Final Predictions ---")
predictions = XOR_Model.forward(X_data)

for x, y_true, y_pred in zip(X_data, Label, predictions):
    print(f"Input: {x} | True: {y_true} | Predicted: {y_pred[0]:.4f}")



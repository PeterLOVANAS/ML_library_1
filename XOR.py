from dense import Dense
from activation_func import Tanh
from loss import mse, mse_prime , mse_prime_BGD , mse_BGD
import numpy as np
import matplotlib.pyplot as plt

X = np.reshape([[0,0], [0,1], [1,0], [1,1]] , (4,2,1))  # (num_sample , width , height)
Y = np.reshape([[0] , [1] , [1] , [0]] , (4,1,1))
# Reshape for making the dataset into column vector

model = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]

modelV2 = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]

# Mini-batch gradient descent
epochs = 100
lr = 0.1
batch = 2 # batch size
X_spl = np.array_split(X, batch)
Y_spl = np.array_split(Y, batch)

"""
Idea
We will calculate square error based on each sample. 
Then, find the mean with respect to the number of batches.
After that, we will get the output_gradient of that batch, this will be use for backprop. algorithm

"""



def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


print("Model V1") # Using MBGD (Mini-batch gradient descent)
for e in range(epochs):
    errors = []
    for x , y in zip(X_spl , Y_spl): # Batch loop
        error_sum = 0
        for x1 , y1 in zip(x,y): # Inside each batch loop
            # Forward propagation
            output = x1
            for layers in model:
                output = layers.forward(output)
            error_sum += mse_prime_BGD(y1 , output)
            errors.append(mse_BGD(y1 , output))

        grad_y = error_sum / batch # Average output gradient
        for layers in reversed(model):
            grad_y = layers.backward(grad_y , lr)



    epoch_err = sum(errors) / len(errors)
    print(f"epochs {e+1} : error = {epoch_err[0][0]:.2f}")





print("Model V2")  # Using SGD
for e in range(epochs):
    error = 0
    for x , y in zip(X,Y):
        output = x
        for layers in modelV2:
            output = layers.forward(output)

        error += mse(y , output)
        grad_y = mse_prime(y, output)
        for layers in reversed(modelV2):
            grad_y  = layers.backward(grad_y,lr)

    error /= len(X)
    print(f"epochs {e+1} : error = {error:.2f}")


points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(model, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

points2 = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(modelV2, [[x], [y]])
        points2.append([x, y, z[0,0]])

points2 = np.array(points2)


fig = plt.figure()
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")
ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c=points2[:, 2], cmap="winter")
ax1.set_title("Model V1")
ax2.set_title("Model V2")
plt.show()







"""
for e in range(epochs):
    error = 0
    pred_lst =[]

    for x ,y in zip(X, Y):
        # Forward propagation
        output = x
        for layers in model:
            output = layers.forward(output)
        


    pred_vec = np.array(pred_lst).reshape((4, 1 ,1))
    error += mse(Y , pred_vec)
    grad_Y = mse_prime(Y, pred_vec)
    print(pred_vec)
    for layers in reversed(model):
        grad_Y = layers.backward(grad_Y , lr)
        # Training occurs in parameters like weight or bias which is in Dense() while Tanh() just create the gradient vector for passing to its previous layer

    print(f"epochs {e+1} : error = {error:.2f}") # e+1 comes from the point that when we run loops we start with 0
"""




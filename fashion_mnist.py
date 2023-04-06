import numpy as np
import matplotlib.pyplot as plt
import pathlib
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from dense import Dense
from activation_func import Sigmoid , Softmax
from convolution import Convolution, Reshape
from loss import binary_cross_entropy , binary_cross_entropy_prime, mse , mse_prime
from keras.datasets import fashion_mnist
from alive_progress import alive_bar
from skimage.measure import block_reduce
from utility import MaxPooling2D
from activation_func import ReLU, Tanh



def get_fashion_mnist():
    (train_images, train_labels) , (test_images , test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype("float32") / 255 # Normalized
    test_images = test_images.astype("float32") / 255 # Normalized
    train_images = np.reshape(train_images, (train_images.shape[0],1, train_images.shape[1] , train_images.shape[2]))
    test_images = np.reshape(test_images, (test_images.shape[0],1, test_images.shape[1] , test_images.shape[2]))
    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]
    train_labels = train_labels.reshape(len(train_images) , 10 , 1)
    test_labels = test_labels.reshape(len(test_images) , 10 , 1)
    return train_images , test_images , train_labels , test_labels

train_images_1 , test_images_1 , train_labels_1 , test_labels_1 = get_fashion_mnist()
train_images = train_images_1[:10000]
train_labels = train_labels_1[:10000]
test_images = test_images_1[:1000]
test_labels = test_labels_1[:1000]
print("finished loading data")




modelV2 = [
    Convolution((1,28,28) , 3, 5 ),
    ReLU(),
    MaxPooling2D((2,2) , 2),
    Convolution((1,13 ,13) , 3, 5),
    ReLU(),
    MaxPooling2D((2,2) ,2),
    Reshape((5,5,5),(5*5*5 ,1)),
    Dense(5*5*5 , 100),
    ReLU(),
    Dense(100 , 10),
    Softmax()
]

modelV1 = [
    Convolution((1,28,28) , 3, 5 ),
    ReLU(),
    MaxPooling2D((2,2) , 2),
    Convolution((1,13 ,13) , 3,5),
    ReLU(),
    Reshape((5,11,11), (5*11*11 , 1)),
    Dense(5*11*11, 100),
    Sigmoid(),
    Dense(100 , 10),
    Softmax()
]

epochs = 50
lr = 0.001


# training
for e in range(epochs):
    error = 0
    num_sample = train_labels.shape[0]
    tracking = 1
    nr_correct = 0
    with alive_bar(total = num_sample , title =f"Epoch {e+1}" , theme = "smooth" ) as bar:
        for x , y in zip(train_images , train_labels):
            # Forward propagation
            output = x
            for layers in modelV1:
                output = layers.forward(output)

            error += mse(y, output)
            nr_correct += int(np.argmax(output) == np.argmax(y))


            # Backward propagation
            output_gradient = mse_prime(y, output)
            for layers in reversed(modelV1):
                if isinstance(layers, MaxPooling2D) == True:
                     output_gradient = layers.backward(output_gradient)
                else:
                    output_gradient = layers.backward(output_gradient , lr)


            bar()



    error /= train_labels.shape[0]
    print(f"epochs {e+1} : error = {error:.2f} , accuracy : {(nr_correct / train_labels.shape[0]) * 100 : .3f} %")


# testing

nr_correct_test = 0
with alive_bar(total = test_labels.shape[0] , title =f"Epoch {e+1}" , theme = "smooth" ) as bar:
    for x ,y in zip(test_images , test_labels):
        output = x
        for layers in modelV1:
            output = layers.forward(output)
        nr_correct_test += int(np.argmax(output) == np.argmax(y))
        bar()


accuracy = (nr_correct_test / test_labels.shape[0] ) * 100
print(f"Test accuracy: {accuracy: .3f}")

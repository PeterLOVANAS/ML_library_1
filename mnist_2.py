import numpy as np
import matplotlib.pyplot as plt
import pathlib
from matplotlib import pyplot as plt

import loss
from dense import Dense
from activation_func import Sigmoid , Softmax
from convolution import Convolution, Reshape
from loss import *
from keras.datasets import mnist
from alive_progress import alive_bar
from model import Sequential
from optimizer import Gradient_Descent
from utility import dataloader
from utility import *

# get the data
def get_mnist():
    (train_images, train_labels) , (test_images , test_labels) = mnist.load_data()
    train_images = train_images.astype("float32") / 255 # Normalized
    test_images = test_images.astype("float32") / 255 # Normalized
    train_images = np.reshape(train_images, (train_images.shape[0],1, train_images.shape[1] , train_images.shape[2])) # match the input of the model (in this case, convolution)
    test_images = np.reshape(test_images, (test_images.shape[0],1, test_images.shape[1] , test_images.shape[2]))
    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]
    train_labels = train_labels.reshape(len(train_images) , 10 , 1)
    test_labels = test_labels.reshape(len(test_images) , 10 , 1)
    return train_images , test_images , train_labels , test_labels


train_images_1 , test_images_1 , train_labels_1 , test_labels_1 = get_mnist()


train_images = train_images_1[:10000]
train_labels = train_labels_1[:10000]

val_images = train_images_1[30001:35001]
val_labels = train_labels_1[30001:35001]

test_images = test_images_1[:1000]
test_labels = test_labels_1[:1000]
print("finished loading data")

# preprocessing
train_list = [train_images , train_labels]
val_list = [val_images , val_labels]

data_train = dataloader(train_list , batch_size= 512 , shuffle= True )
data_val = dataloader(val_list , batch_size=512 , shuffle= True)
print("finished preprocessing")


# Defining the model's architecture

model_V1 = Sequential([
    Convolution((1,28,28) , 3, 5 ),
    Sigmoid(),
    Reshape((5,26,26), (5*26*26 , 1)),
    Dense(5*26*26, 100),
    Sigmoid(),
    Dense(100 , 10),
    Softmax()
])

print("finish defining the model's architecture")

# Training the model
print("start of training")
opt = Gradient_Descent(avg_grad_model=[] ,learning_rate=0.01 , momentum=0.9 , acceleration=True)
model_V1.compile(optimizer=opt , loss= loss.mse , metric=["accuracy"])
hist = model_V1.fit(data= data_train , epochs = 2 , validation_data=data_val)
print(hist) 



# shift+r => to run this code on a python interactive window 





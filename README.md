In this project, we will try to perform an Image Classification task using an _Artificial Neural Network_. We will try to investigate phenomena such as _underfitting_ and _overfitting_, and will come up with some solutions to these problems. The aim of this project isn't necessarily to obtain the highest accuracy, but make sure the model maintains similar levels of performances with both training and unseen data. We use Tensorflow in this project.

You can continue with this post or jump right to the notebook right [here](https://github.com/oppongk23/Image-Classification-ANN/blob/main/ANN_Assignment.ipynb). 

#### The Dataset
The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is used in this project.The dataset is a collection of about 60,000 small images. Each image is an RGB image with a size of 32x32. The dataset is also divided into two sets: a training set and a test set. The training data comprises 50,000 images divided into batches of 10,000 images and the test set comprises 10,000 images. There are 10 different classes in the dataset and each training batch contains 1,000 images from each class.

In the course of the project the data is normalized and converted to grayscale to aid a faster training process and to reduce the computational load.


#### Importing the Dependencies
We begin by importing the tensorflow and keras libraries along with some other modules in those libraries.

```
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout
```


#### Dataset Preparation
We then download the dataset and normalize it, then change it to grayscale. By doing this, we take away the 3 RGB channels and replace it with only one channel.

```
# Downloading the cifar10 dataset 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Changing the data type of training and testing data and normalizing them
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0


# Changing the images to grayscale
def grayscale(data, dtype='float32'):
  # luma coding weighted average in video systems
  r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
  rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
  # add channel dimension
  rst = np.expand_dims(rst, axis=3)
  return rst

X_train_gray = grayscale(x_train)
X_test_gray = grayscale(x_test)

```

After plotting, we have the image below
```
# Plotting the first 10 images again to ensure the mages are in grayscale
display_images = X_train_gray.reshape(-1, 32,32)
fig, axes = plt.subplots(1, 10, figsize = (30, 10))
for img, ax in zip(display_images[:10], axes):
  ax.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
```
![Grayscale Images](https://github.com/oppongk23/Image-Classification-ANN/blob/main/Images/cifar10_gray.png "Grayscale Images")


#### Building the first model
We can then construct an ANN with an input layer, 8 hidden layers and an output layer. Each of the 8 hidden layers will have 512 units (perceptrons) with the ReLU activation funciton. ReLU is a non-linear function that takes the maximum argument between a value passed to it and 0. 
The final layer will use a softmax function to output a series of probabilities relative to each class. The class with the highest probability is chosen as the class of the input image.

The training and validation sets will be split in an 80:20 ratio.

Then we can go ahead and train our model

```
# Creating the model with 8 hidden layers aside the input and output layers
model = keras.Sequential([keras.layers.Flatten(input_shape = (32, 32)), 
                          keras.layers.Dense(512, activation="relu"),
                          keras.layers.Dense(512, activation="relu"),
                          keras.layers.Dense(512, activation="relu"), 
                          keras.layers.Dense(512, activation="relu"),
                          keras.layers.Dense(512, activation="relu"),
                          keras.layers.Dense(512, activation="relu"),
                          keras.layers.Dense(512, activation="relu"),
                          keras.layers.Dense(512, activation="relu"),
                          keras.layers.Dense(10, activation= "softmax")])
                          
# Compiling the model with the adam optimizer, cross entropy loss, and accuracy as the metric
model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

# Training the model
training_summary = model.fit(X_train_gray, y_train, epochs=20, batch_size=128, validation_split=0.2)
```


#### Analysing the results
We plot the training and validation losses and accuracies, to see how the model performs on both seen and unseen data.
Below is the plot of the training and validation losses.


![Training and Validation Losses](https://github.com/oppongk23/Image-Classification-ANN/blob/main/Images/model1loss.png "Grayscale Images")

Also, we can take a look at the training and validation accuracies.

![Training and Validation Losses](https://github.com/oppongk23/Image-Classification-ANN/blob/main/Images/model1acc.png "Grayscale Images")



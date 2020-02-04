import os
import numpy as np
import pickle
from sklearn.utils import shuffle
from PIL import Image, ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_cifar_10_data():
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for file_name in os.listdir("data/cifar/train"):
        image = np.array(Image.open("data/cifar/train/" + file_name))
        image = image.reshape((3, 32, 32))
        image = image.astype(np.float64) / 255
        x_train.append(image)
        y_train.append(classes.index(file_name.split(".")[0].split("_")[1]))

    for file_name in os.listdir("data/cifar/test"):
        image = np.array(Image.open("data/cifar/test/" + file_name))
        image = image.reshape((3, 32, 32))
        image = image.astype(np.float64) / 255
        x_test.append(image)
        y_test.append(classes.index(file_name.split(".")[0].split("_")[1]))

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def generation_data(batch_size, x_train, y_train):
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    for i in range(0, len(x_train), batch_size):
        if i + batch_size < len(x_train):
            yield x_train[i: i + batch_size], y_train[i: i + batch_size]
        else:
            yield x_train[i: len(x_train)], y_train[i: len(y_train)]
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from os import listdir
from os.path import isfile, join
from PIL import Image
import keras.api._v2.keras as keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

IMAGES_PATH = './images/mnist/'
PREDICTION_IMAGES_PATH  = './images/predict/'
MODEL_PATH = './models/mnist.hdf5'
DIGITS = list(range(0, 9))

def load_images(path):
    images = []
    labels = []
    filenames = os.listdir(path)

    for filename in filenames:
        image = imread(path + filename)
        image = image / 255

        images.append(image)
        labels.append(filename[0])       

    np.expand_dims(images, axis=3)

    return np.array(images), np.array(labels)

def train_model(images_train, images_test, labels_train, labels_test):

    epochs = 50
    batch_size = 100

    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)

    model = Sequential([
        Conv2D(32, kernel_size=(3,3), input_shape=(28, 28, 1), padding='same', activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        Dropout(0.5),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(images_train, to_categorical(labels_train), epochs=epochs, batch_size=batch_size, validation_data=(images_test, to_categorical(labels_test)), callbacks=[es])

    return model

def plot_accuracy(model):
    history = model.history
  
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
  
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

images, labels = load_images(IMAGES_PATH)
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2)
model = train_model(images_train, images_test, labels_train, labels_test)

plot_accuracy(model)

model.save(MODEL_PATH)

import os
import matplotlib.pyplot as plt
import numpy as np
import gc 
from sklearn.model_selection import train_test_split
from skimage.io import imread
import keras.api._v2.keras as keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

IMAGES_PATH = './images/quickdraw/'
PREDICTION_IMAGES_PATH  = './images/predict/'
MODEL_PATH = './models/sketches.hdf5'
TEMP_IMAGES_FILE = './images.npy'
TEMP_LABELS_FILE = './labels.npy'
MAX_IMAGES = 10000
WIDTH = 28
HEIGHT = 28
NUM_CLASSES = 130

def load_images(path):
    images = []
    labels = []
    filenames = os.listdir(path)[0:NUM_CLASSES]
    
    for i, filename in enumerate(filenames):
        print(filename)
        data = np.load(path + filename, mmap_mode='r')[0:MAX_IMAGES]
        data = np.reshape(data, (data.shape[0], WIDTH, HEIGHT))
        
        for image in data:
            images.append(image)
            labels.append(i)       

        del data
        gc.collect()
    
    print('Pre-processing images')
    images = np.array(images).astype('float32') / 255.
    labels = np.array(labels)

    print('Expanding image dimensions')
    np.expand_dims(images, axis=3)
    
    #print('Saving pre-processed images and labels')
    #np.save(TEMP_IMAGES_FILE, images)
    #np.save(TEMP_LABELS_FILE, labels)
    
    return images, labels

def train_model(images_train, images_test, labels_train, labels_test):
    epochs = 50
    batch_size = 64
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)

    #model = Sequential([
    #    Conv2D(32, kernel_size=(3,3), input_shape=(WIDTH, HEIGHT, 1), padding='same', activation='relu'),
    #    MaxPooling2D(pool_size=(2,2)),
    #    Conv2D(64, kernel_size=(3,3), activation='relu'),
    #    MaxPooling2D(pool_size=(2,2)),
    #    Conv2D(64, kernel_size=(3,3), activation='relu'),
    #    Dropout(0.5),
    #    Flatten(),
    #    Dense(NUM_CLASSES, activation='relu'),
    #    Dropout(0.1),
    #    Dense(NUM_CLASSES, activation='softmax')
    #])

    model = Sequential([
        Conv2D(input_shape=(WIDTH, HEIGHT, 1), kernel_size=5, filters=32, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(kernel_size=3, filters=32, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(kernel_size=3, filters=64, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2, strides=2),
        Flatten(),
        Dense(units=512, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
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
#images = np.load(TEMP_IMAGES_FILE, mmap_mode='r')
#labels = np.load(TEMP_LABELS_FILE, mmap_mode='r')
    
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2)
model = train_model(images_train, images_test, labels_train, labels_test)

plot_accuracy(model)

model.save(MODEL_PATH)

import os
import numpy as np
from skimage.io import imread, imshow
import keras.api._v2.keras as keras
from matplotlib import pyplot as plt

PREDICTION_IMAGES_PATH  = './images/predict/'
MODEL_PATH = './models/model.hdf5'
DIGITS = list(range(0, 10))

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

    
model = keras.models.load_model(MODEL_PATH)

predicition_images, prediction_labels = load_images(PREDICTION_IMAGES_PATH)
predictions = model.predict(predicition_images, verbose=0)

for i, prediction in enumerate(predictions):
    print('\r\n----> ' + prediction_labels[i])    
    z = list(zip(prediction, DIGITS))
    
    sorted_prediction = sorted(z, reverse=True)

    for item in sorted_prediction:
        if (item[0] > 0.01):
            print(f'{item[1]}: {round(item[0]*100)}%')

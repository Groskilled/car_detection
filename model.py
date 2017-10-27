import numpy as np
from pylab import *
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.models import *
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D

def Model():

    model = Sequential()
    model.add(Conv2D(64, (7,7), strides=(2,2), activation='relu', input_shape=(64,64,3)))

    model.add(Conv2D(192, (3, 3), strides=(2,2), activation='relu'))

    model.add(Conv2D(128, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(256, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), strides=(3,3), activation='relu', padding='same'))
    
    model.add(Conv2D(128, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), strides=(3,3), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    #model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


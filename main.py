from keras.preprocessing.image import ImageDataGenerator
from model import Model
import sys
import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math

def get_img_parts(img):
    imgs = []
    for i in range(0,240, 120):
        for j in range(0,1278, 213):
            tmp = cv2.resize(img[i:i+120,j:j+213], (64,64))/255
            imgs.append(tmp)
            tmp = cv2.resize(img[i:i+120,j+107:j+320], (64,64))/255
            imgs.append(tmp)
            tmp = cv2.resize(img[i+60:i+180,j+107:j+320], (64,64))/255
            imgs.append(tmp)
            tmp = cv2.resize(img[i+60:i+180,j:j+213], (64,64))/255
            imgs.append(tmp)
            #plt.imshow(tmp)
            #plt.show()
    return imgs

def train_model():
    batch_size = 20
    model = Model()
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory('train_images',target_size=(64, 64),batch_size=batch_size,class_mode='binary')
    validation_generator = test_datagen.flow_from_directory('val_images',target_size=(64, 64),batch_size=batch_size,class_mode='binary')
    model.fit_generator(train_generator,epochs=15, steps_per_epoch=9553 // batch_size, validation_data=validation_generator, validation_steps=6367 // batch_size)
    model.save_weights('weights_1.h5')

#train_model()
#sys.exit()
model = Model()
model.load_weights('weights_1.h5')
images = glob.glob('test_images/*.jpg')
pos_lst = []
for idx,fname in enumerate(images):
    img = mpimg.imread(fname)
    img = img[400:640,1:1279]
    #cv2.line(img, (0,120),(1278,120),(255,0,0),2)
    #cv2.line(img, (213,0),(213,240),(255,0,0),2)
    #cv2.line(img, (426,0),(426,240),(255,0,0),2)
    #cv2.line(img, (639,0),(639,240),(255,0,0),2)
    #cv2.line(img, (852,0),(852,240),(255,0,0),2)
    #cv2.line(img, (1065,0),(1065,240),(255,0,0),2)
    #cv2.line(img, (1278,0),(1278,240),(255,0,0),2)
    #print(img.shape)
    #for i in range(0,240, 80):
    #    for j in range(0,1280, 80):
    #        tmp = img[i:i+80,j:j+80]
    #        imgs.append(cv2.resize(tmp,(64,64))/255)
    imgs = np.array(get_img_parts(img))
    ret = np.where(model.predict(imgs) > 0.95)[0]
    for p in ret:
        plt.imshow(imgs[p])
        plt.show()

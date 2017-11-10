from keras.preprocessing.image import ImageDataGenerator
from model import Model
import sys
import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

def get_img_parts(img, sizex):
    imgs = []
    x = img.shape[1]
    for i in range(0,x, int(sizex * 0.3)):
        if i + sizex > x + sizex:
            break
        tmp = img[:,i:i+sizex]
        imgs.append(cv2.resize(tmp,(64,64))/255)
    return imgs

def heat_map(heatmap, boxes, x, box_size, y_top, y_bot):
    for box in boxes:
        heatmap[y_top:y_bot, box * x: box * x + box_size] += 1
    return heatmap

def train_model():
    batch_size = 32
    model = Model(pretrained=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.5,zoom_range=0.2,horizontal_flip=True, rotation_range=0.5,width_shift_range=0.3,height_shift_range=0.3)
    train_generator = train_datagen.flow_from_directory('train_images',target_size=(64, 64),batch_size=batch_size,class_mode='binary')
    validation_generator = test_datagen.flow_from_directory('val_images',target_size=(64, 64),batch_size=batch_size,class_mode='binary')
    model.fit_generator(train_generator,epochs=25, steps_per_epoch=9553 // batch_size, validation_data=validation_generator, validation_steps=6367 // batch_size)
    model.save_weights('weights_fcn.h5')

def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img

#train_model()
#sys.exit()
model = Model()
model.load_weights('weights_fcn.h5')
images = glob.glob('test_images/*.jpg')
pos_lst = []
for idx,fname in enumerate(images):
    img = mpimg.imread(fname)
    tmp = img[400:656]

    img_size = 32 * 2
    
    imgs1 = np.array(get_img_parts(tmp[:img_size], img_size))
    cars1 = np.where(model.predict(imgs1) > 0.5)[0]
    
    heat = np.zeros_like(tmp[:,:,0]).astype(np.float)
    heat = heat_map(heat, cars1, int(img_size*0.3), img_size, 0, img_size)

    heat = np.clip(heat,0,255)
    labels = label(heat)

    draw_img = draw_labeled_bboxes(np.copy(tmp), labels)
    plt.imshow(draw_img)
    plt.show()

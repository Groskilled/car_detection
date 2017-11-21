from skimage.feature import hog
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import sys
import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

def get_img_parts(img, sizex):
    imgs = []
    x = img.shape[1]
    for i in range(0,x, int(sizex * 0.5)):
        if i + sizex > x:
            break
        tmp = img[:,i:i+sizex]
        imgs.append(cv2.resize(tmp,(64,64)))
    return imgs


def get_hog_features(img, orient, pix_per_cell, cell_per_block,vis=False, feature_vec=True):
    features = hog(img, orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   transform_sqrt=False,
                   visualise=vis, feature_vector=feature_vec)
    return features

def extract_features(image, orient=11,pix_per_cell=16, cell_per_block=2):
    feature_image = np.copy(image)
    hog_features = []
    for channel in range(feature_image.shape[2]):
        hog_features.append(get_hog_features(feature_image[:,:,channel],orient, pix_per_cell, cell_per_block,feature_vec=True))
    hog_features = np.ravel(hog_features)
    return np.array(hog_features)

def heat_map(heatmap, boxes, x, box_size, y_top, y_bot):
    for box in boxes:
        heatmap[y_top:y_bot, box * x: box * x + box_size] += 1
    return heatmap

def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img

def get_heat(img):
    tmp = img[400:]

    heat = np.zeros_like(tmp[:,:,0]).astype(np.float)
        
    small_imgs1 = np.array(get_img_parts(tmp[:64], 64))
    small_imgs2 = np.array(get_img_parts(tmp[16:80], 64))
    small_imgs1 = [extract_features(x) for x in small_imgs1]
    small_imgs2 = [extract_features(x) for x in small_imgs2]
    
    little_imgs1 = np.array(get_img_parts(tmp[:96], 96))
    little_imgs2 = np.array(get_img_parts(tmp[32:128], 96))
    little_imgs1 = [extract_features(x) for x in little_imgs1]
    little_imgs2 = [extract_features(x) for x in little_imgs2]

    mid_imgs1 = np.array(get_img_parts(tmp[:128], 128))
    mid_imgs2 = np.array(get_img_parts(tmp[30:158], 128))
    mid_imgs1 = [extract_features(x) for x in mid_imgs1]
    mid_imgs2 = [extract_features(x) for x in mid_imgs2]

    big_imgs1 = np.array(get_img_parts(tmp[:196], 256))
    big_imgs2 = np.array(get_img_parts(tmp[64:196+64], 256))
    big_imgs1 = [extract_features(x) for x in big_imgs1]
    big_imgs2 = [extract_features(x) for x in big_imgs2]

    for model in models:
        small_cars1 = np.where(model.predict(small_imgs1) >= 0.5)[0]
        small_cars2 = np.where(model.predict(small_imgs2) >= 0.5)[0]
        heat = heat_map(heat, small_cars1, int(64*0.5), 64, 0, 64)
        heat = heat_map(heat, small_cars2, int(64*0.5), 64, 16, 80)
        
        little_cars1 = np.where(model.predict(little_imgs1) >= 0.5)[0]
        little_cars2 = np.where(model.predict(little_imgs2) >= 0.5)[0]
        heat = heat_map(heat, little_cars1, int(96*0.5), 96, 0, 96)
        heat = heat_map(heat, little_cars2, int(96*0.5), 96, 32, 128)

        mid_cars1 = np.where(model.predict(mid_imgs1) >= 0.6)[0]
        mid_cars2 = np.where(model.predict(mid_imgs2) >= 0.6)[0]
        heat = heat_map(heat, mid_cars1, int(128*0.5), 128, 0, 128)
        heat = heat_map(heat, mid_cars2, int(128*0.5), 128, 30, 158)
        
        big_cars1 = np.where(model.predict(big_imgs1) > 0.9)[0]
        big_cars2 = np.where(model.predict(big_imgs2) > 0.9)[0]
        heat = heat_map(heat, big_cars1, int(196*0.5), 196, 0, 196)
        heat = heat_map(heat, big_cars2, int(196*0.5), 196, 64, 196+64)
        #heat[heat < 2] = 0

    heat[heat < 2] = 0
    heat = np.clip(heat,0,255)
    return heat


def process_img(img):
    imgs_list.append(img)
    if len(imgs_list) > 7:
        imgs_list.pop(0)
    heat_list.append(get_heat(img))
    if len(heat_list) > 7:
        heat_list.pop(0)
    heat = np.array(sum(heat_list))
    heat[heat < 4] = 0
    #return img

    labels = label(heat)
    draw_img = draw_labeled_bboxes(np.copy(img[400:]), labels)
    img[400:] = draw_img
    return img

#car_images = glob.glob('train_images/vehicles/*.png')
#noncar_images = glob.glob('train_images/non-vehicles/*.png')
#y = np.hstack((np.ones(len(car_images)), np.zeros(len(noncar_images))))
#car_images = [extract_features(mpimg.imread(x)) for x in car_images]
#noncar_images = [extract_features(mpimg.imread(x)) for x in noncar_images]
#x = np.vstack((car_images, noncar_images))
#scaler = preprocessing.StandardScaler().fit(x)
#scaler.transform(x)

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#
#print('mdr')
#model = LinearSVC() 
#model.fit(x_train,y_train)
#print(model.score(x_test, y_test))
#pickle.dump(model, open('svc_save.txt', 'wb'))
###
##
#model = GradientBoostingClassifier()
#model.fit(x_train,y_train)
#print(model.score(x_test, y_test))
#pickle.dump(model, open('boost_save.txt', 'wb'))
##
#
#params = {
#    'objective': 'binary',
#    'boosting_type': 'gbdt',
#    'metric': 'binary_logloss',
#}
#
#lgb_train = lgb.Dataset(x_train, y_train)
#lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
#
#gbm = lgb.train(params,lgb_train,num_boost_round=500,valid_sets=lgb_eval,early_stopping_rounds=5)
#print(model.score(x_test, y_test))
#gbm.save_model('lgbm.txt')
#
#sys.exit()

models = []
models.append(lgb.Booster(model_file='lgbm.txt'))
#models.append(pickle.load(open('svc_save.txt', 'rb')))
#models.append(pickle.load(open('boost_save.txt', 'rb')))

imgs_list = []
heat_list = []
video_output = 'result_long.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_img)
white_clip.write_videofile(video_output, audio=False)

sys.exit()
#
#models = []
#models.append(lgb.Booster(model_file='lgbm.txt'))
##models.append(pickle.load(open('svc_save.txt', 'rb')))
#models.append(pickle.load(open('boost_save.txt', 'rb')))
images = glob.glob('test_images/*.jpg')
for idx,fname in enumerate(images):
    img = mpimg.imread(fname)
    tmp = img[400:]

    heat = np.zeros_like(tmp[:,:,0]).astype(np.float)
        
    small_imgs1 = np.array(get_img_parts(tmp[:64], 64))
    small_imgs2 = np.array(get_img_parts(tmp[16:80], 64))
    small_imgs1 = [extract_features(x) for x in small_imgs1]
    small_imgs2 = [extract_features(x) for x in small_imgs2]
    
    little_imgs1 = np.array(get_img_parts(tmp[:96], 96))
    little_imgs2 = np.array(get_img_parts(tmp[32:128], 96))
    little_imgs1 = [extract_features(x) for x in little_imgs1]
    little_imgs2 = [extract_features(x) for x in little_imgs2]

    mid_imgs1 = np.array(get_img_parts(tmp[:128], 128))
    mid_imgs2 = np.array(get_img_parts(tmp[30:158], 128))
    mid_imgs1 = [extract_features(x) for x in mid_imgs1]
    mid_imgs2 = [extract_features(x) for x in mid_imgs2]

    for model in models:
        small_cars1 = np.where(model.predict(small_imgs1) >= 0.3)[0]
        small_cars2 = np.where(model.predict(small_imgs2) >= 0.3)[0]
        heat = heat_map(heat, small_cars1, int(64*0.5), 64, 0, 64)
        heat = heat_map(heat, small_cars2, int(64*0.5), 64, 16, 80)
        
        little_cars1 = np.where(model.predict(little_imgs1) >= 0.3)[0]
        little_cars2 = np.where(model.predict(little_imgs2) >= 0.3)[0]
        heat = heat_map(heat, little_cars1, int(96*0.5), 96, 0, 96)
        heat = heat_map(heat, little_cars2, int(96*0.5), 96, 32, 128)

        mid_cars1 = np.where(model.predict(mid_imgs1) >= 0.3)[0]
        mid_cars2 = np.where(model.predict(mid_imgs2) >= 0.3)[0]
        heat = heat_map(heat, mid_cars1, int(128*0.5), 128, 0, 128)
        heat = heat_map(heat, mid_cars2, int(128*0.5), 128, 30, 158)

    heat[heat < 2] = 0
    heat = np.clip(heat,0,255)
    #plt.imshow(heat)
    #plt.show()
    labels = label(heat)

    draw_img = draw_labeled_bboxes(np.copy(tmp), labels)
    #plt.imshow(draw_img)
    #plt.show()

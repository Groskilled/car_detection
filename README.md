**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/sliding_window.png
[image3]: ./output_images/heat_map.png
[image4]: ./output_images/labeled.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Histogram of Oriented Gradients (HOG)

The code for this step is contained in the get_hog_features() function at line 31.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and went for the one that gave me the best validation score.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained many classifiers (NN, SVC, XGBC, LGBM ...) and I had a lot of problem with those. They achieved a good score (97% and more) on my validation set, but when used on the test images it did not performed as well. I tried combinations and stuff like that and in the end I went for lgbm with a threshold of 30%.

###Sliding Window Search

I decided to use only the bottom of the image, starting a 400px since above there is little chance to see a vehicle. Since I need more than one size for the windows due to different vehicle sizes, I created a function: get_img_parts at line 20, that return parts of the image according to the desired input size and rescaled to 64x64.

![alt text][image2]

I then get prediction on every part of the image I created for different size with an overlap of 50% on X axis and 75% on Y axis. Then I create a heatmap of the image using a threshold of 2 (everything that has been marked 1 time or less as a car is discarded). I end up with this:
![alt text][image3]

And finally I get labels from the heatmap and put in on the original image.
![alt_text][image4]

---

### Video Implementation

The output video is in the output_images folder.

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. To minimize the number of false positives I am recording the heatmaps over 7 images, add them and apply a threshold. I the return the image with the boxes drawn on it.  

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I started with a vgg16 but it performed poorly for a long training time and using data augmentation. I then moved to something simpler, a SVC. Since I tried SVC, I though as well try with an xgboost and lgbm. SVC and XGBoost had a lot of false positive. I then tried to run those 3 models on the image and put a higher threshold. But this did not do the trick, here again a lot of false positive and the white vehicle was not detected most of the time. I wanted to try implementing a yolo net but I did not find a pretrained version and with my computer I cannot aford to train that kind of model, it would take ages. So I kept my lgbm since it was the best performing one, searched for the right threshold and then did the average over multiple images.

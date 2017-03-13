**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1.0]: ./CarND-Vehicle-Detection-master/output_images/vehicles_test_examples.png
[image1.1]: ./CarND-Vehicle-Detection-master/output_images/non_vehicles_test_examples.png
[image2]: ./CarND-Vehicle-Detection-master/output_images/HOG_example.png
[image3]: ./CarND-Vehicle-Detection-master/output_images/scale_1.4.png
[image4]: ./CarND-Vehicle-Detection-master/output_images/search_windows.png
[image5]: ./CarND-Vehicle-Detection-master/output_images/six_frames_with_heatmaps.png
[image6]: ./CarND-Vehicle-Detection-master/output_images/labels_heatmap_and_bboxes.png
[video1]: ./CarND-Vehicle-Detection-master/output_images/final_output_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell 3 of the Jupyter Notebook `writeup.ipynb`  

I started by reading in all the `vehicle` and `non-vehicle` images (cell 1).  Here is an example of one of each of the `vehicle` ...

![alt text][image1.0]

and `non-vehicle` classes ....

![alt text][image1.1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example (from cell 12) using the `RGB` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8)` and `cells_per_block=(2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters (cell 13) :
```python
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# Train 1 RGB, orient=6, hog_channel=0, spatial_size=(16, 16), hist_bins=16 : Test Accuracy == 0.97
# Train 2 YCrCb, orient=6, hog_channel=0, spatial_size=(16, 16), hist_bins=16 : Test Accuracy == 0.96
# Train 3 YCrCb, orient=9, hog_channel=0, spatial_size=(16, 16), hist_bins=16 : Test Accuracy == 0.975
# Train 4 YCrCb, orient=9, hog_channel="ALL", spatial_size=(16, 16), hist_bins=16 : Test Accuracy == 0.985
# Train 5 YCrCb, orient=9, hog_channel="ALL", spatial_size=(32, 32), hist_bins=32 : Test Accuracy == 0.995
orient = 9  # HOG orientation angles
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"  # aka 1st feature channel. Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM (cell 13) using the above parameters and recorded the Test Accuracy until I could get close to 1.0. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search a Region Of Interest (cell 7) at scale 1.4 (cell 9) :

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./CarND-Vehicle-Detection-master/output_images/final_output_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video (cell 37).  From the positive detections I created a heatmap and then thresholded that map (cell 38) to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image6]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
* I closely followed the [lessons code](https://www.youtube.com/watch?v=P2zwrTM8ueA&feature=youtu.be) and experimented with the different parameters.

* Heat threshold approach (cell 40) on single images was OK but was difficult to tune correctly. To improve this I would need to integrate the heatmaps over 5 to 10 frames, followed by applying the threshold :
```python
# Apply threshold to help remove false positives
heat = apply_threshold(heat, 1)  # shows far-left + far-right false positives
# heat = apply_threshold(heat, 2)  # shows          far-right false positive
# heat = apply_threshold(heat, 57)  # shows                no false positives, but unacceptable bounding box!
```

* While this was an interesting Computer Vision project, it shows just how fragile this 'traditional' approach is aka it does not generalise well.
If I had more time I'd love to implement the Deep Learning "Single Shot Multi-Box Detector" [model](https://arxiv.org/pdf/1512.02325.pdf) approach.
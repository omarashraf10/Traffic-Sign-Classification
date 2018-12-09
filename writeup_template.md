# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test images/test1.png "Traffic Sign 1"
[image5]: ./test images/test2.png "Traffic Sign 2"
[image6]: ./test images/test3.png "Traffic Sign 3"
[image7]: ./test images/test4.png "Traffic Sign 4"
[image8]: ./test images/test5.png "Traffic Sign 5"
[image9]: ./test images/test6.png "Traffic Sign 6"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/omarashraf10/Traffic-Sign-Classification/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is i visualized 6 random images from the training data set by using matplotlib library

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the images by using cv.normalize function 

Here is an example of a traffic sign image before and after normalizing.

![before][image2]



![after][image3]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| dropout    |            |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16      									|
| RELU					|												|
| dropout    |            |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten        |      Output = 400        |
| Fully connected		| Output = 120       									|
| RELU					|												|
| Fully connected		| Output = 84       									|
| RELU					|												|
| Fully connected		| Output = 10       									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a model of leNet 5 with small changes like adding droupout with keep prob and adjusting it to take 3 channel 

images , i initialized the weights of filters with tf.random_normal function and biases with tf.zeros function,

i used batch size equal to 128 , number of EPOCHS equal to 80 epoch , learning rate equal to 0.001 and keep prob equal to 0.7 ,

then i ran the model in a tensorflow session

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9
* validation set accuracy of 94 % 
* test set accuracy of 92.9 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

i started with the leNet 5 architecture in the lessons because it performed well in the MNIST data set in the quiz of the prevoius lesson 

* What were some problems with the initial architecture?

the validation accuracy was littel low , it was about 89 % 

* How was the architecture adjusted and why was it adjusted?

i adjusted it by :-

1- changing the initialization of weights of the filter from tf.truncated_normal to tf.random_normal .
2- adding dropout to avoid overfitting the training set with keep_prob 0.7 .
3- changing the number of epochs to 70 epoch.


* Which parameters were tuned? How were they adjusted and why?
1- number of epochs wasn't epochs so i make the 70  epoch.
2- i tryed some other values for learning rate and catch size but i realized that the initial value of them(learning rate =0.001 , Batch_Size =128) were good 
* What are some of the important design choices and why were they chosen?

i used the requalization technique (dropout) in training the training set to prevent the model from overfitting and perform well in 

images that never see before.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

The fourth image might be difficult to classify because the model can see the number 20 or 30 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Speed limit (70km/h) 									| 
| Roundabout mandatory     			| Turn right ahead 										|
| Stop					| Stop											|
| Speed limit (30km/h)      		| Speed limit (20km/h)					 				|
| Speed limit (120km/h)			| Roundabout mandatory     							|
| Go straight or right			| Go straight or right      							|

The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%. 

in the second image the model predicted the image to be :

-Roundabout mandatory - with percentage of :10.209023
-Turn right ahead - with percentage of :10.048528

it was too close from the correct prediction which is Turn right ahead ,

same thing happend in the fourth image .

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a Speed limit (70km/h) (probability of 19.948889 %), and that wa a correct prediction .

Top 5 Softmax Probabilities For The Image Are :
1st prediction is :Speed limit (70km/h) - with percentage of :19.948889
2st prediction is :Speed limit (20km/h) - with percentage of :14.698641
3st prediction is :Speed limit (30km/h) - with percentage of :12.441849
4st prediction is :Road work - with percentage of :2.061665
5st prediction is :Speed limit (50km/h) - with percentage of :2.023311

For the second image the model was too close from the correct prediction which is Turn right ahead but it is gave a larger prob to Roundabout mandatory 

Top 5 Softmax Probabilities For The Image Are :
1st prediction is :Roundabout mandatory - with percentage of :10.209023
2st prediction is :Turn right ahead - with percentage of :10.048528
3st prediction is :Ahead only - with percentage of :5.51649
4st prediction is :Beware of ice/snow - with percentage of :1.9634647
5st prediction is :Keep left - with percentage of :0.9736037

For the third image, the model is relatively sure that this is a Stop (probability of 25.375076 %), and that wa a correct prediction .

Top 5 Softmax Probabilities For The Image Are :
1st prediction is :Stop - with percentage of :25.375076
2st prediction is :No entry - with percentage of :9.202305
3st prediction is :No vehicles - with percentage of :2.753407
4st prediction is :Speed limit (20km/h) - with percentage of :2.1177557
5st prediction is :Speed limit (30km/h) - with percentage of :0.2108728

For the fourth image the model was too close from the correct prediction which is Speed limit (30km/h) but it is gave a larger prob to Speed limit (20km/h)

Top 5 Softmax Probabilities For The Image Are :
1st prediction is :Speed limit (20km/h) - with percentage of :13.443307
2st prediction is :Speed limit (30km/h) - with percentage of :10.351788
3st prediction is :Speed limit (50km/h) - with percentage of :9.01507
4st prediction is :Speed limit (100km/h) - with percentage of :5.1375537
5st prediction is :Speed limit (80km/h) - with percentage of :3.932359

For the fifth image the model predicted the image wrong 

Top 5 Softmax Probabilities For The Image Are :
1st prediction is :Roundabout mandatory - with percentage of :7.067654
2st prediction is :Dangerous curve to the right - with percentage of :2.8267586
3st prediction is :Speed limit (30km/h) - with percentage of :-0.4415643
4st prediction is :Road work - with percentage of :-1.2951646
5st prediction is :Wild animals crossing - with percentage of :-1.9384843

For the sixth image, the model is relatively sure that this is a Stop (probability of 27.895742 %), and that wa a correct prediction .

Top 5 Softmax Probabilities For The Image Are :
1st prediction is :Go straight or right - with percentage of :27.895742
2st prediction is :Ahead only - with percentage of :16.477861
3st prediction is :Keep right - with percentage of :11.800195
4st prediction is :Roundabout mandatory - with percentage of :7.8315473
5st prediction is :Beware of ice/snow - with percentage of :2.8452482


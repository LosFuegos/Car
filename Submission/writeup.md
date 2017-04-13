#**Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[image2]: ./Images/1.jpg "Grayscaling"
[image3]: ./Images/2.jpg "Recovery Image"
[image4]: ./Images/3.jpg "Recovery Image"
[image5]: ./Images/4.jpg "Recovery Image"
[image6]: ./Images/5.jpg "Normal Image"
[image7]: ./Images/6.jpg "Flipped Image"
[image8]: ./Images/7.jpg "Processed Image"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on Nvidia's Model and consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 114-125)

The model includes ELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 115).

####2. Attempts to reduce overfitting in the model

The model does not contain dropout layers in order to reduce overfitting. But the model was trained and validated on different data sets to ensure that the model was not overfitting (code line 132). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 129).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, recorded extra data only on curves and problem spots, captured data driving the oposite direction, and flipped images and steering data to double the size of the training set.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The one step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. So to improve the driving behavior in these cases, I decided to record recovery data and center lane driving data in problem areas.

First I dumped 70% of images that had near zero steering measurement. I cropped the top 60 pixels and bottom 25 pixels out and reduced the size to (200, 66) to remove noise. I also, converted images to HLS from BGR and added horizontal shifts to my training set only.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 114-125) consisted of a convolution neural network with the following layers One Normalization layer, 5 Convolutional Layers, One Flatten, and 4 Dense layers.



Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0




####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover on its own. These images show what a recovery looks like starting from the edge of track back to center lane:

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

And Here is a Fully Processed Image.

![alt text][image8]

After the collection process, I had 288,684 number of data points.


I finally randomly shuffled the data set and put 25% of the data into a validation set. In drive.py I added a preprocessing method that converted images to HLS and cropped and resized image to (200,66).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 and I used an adam optimizer so that manually training the learning rate wasn't necessary.

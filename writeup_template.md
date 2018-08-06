# **Behavioral Cloning** 

## Writeup Template
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/image1.png "Model Visualization"
[image2]: ./images/image2.jpg "center image"
[image3]: ./images/image3.jpg "right Image"
[image4]: ./images/image4.jpg "left Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py files contained my code to train the model. It involves 3 stages. 
1. Read data from csv path 
2. Assemble all data together
3. Train the model

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

 1. lambda layer to normalize the data 
 2. Cropping layer to remove unwanted data 
 3. 5x5 conv2d layer with 24 filters
 4. 5x5 conv2d layer with 36 filters
 5. 5x5 conv2d layer with 48 filters
 6. 3x3 conv2d layer with 64 filters
 7. 3x3 conv2d layer with 64 filters
 8. Flatten layer
 9. Connected layer with 100 neurons
 10. Connected layer with 50 neurons
 11. Connected layer with 10 neurons
 12. Output layer with 1 neuron

#### 2. Attempts to reduce overfitting in the model

This model used many different sets of data to avoid overfitting.
1. Left Camera
2. Right Camera
3. Center Camera
4. Flipped images with opposite angle value

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. 

#### 4. Appropriate training data

I used 4 different data source to generalize the model.
1. Counter clock wise driving
2. Clock wise driving
3. Smooth curve driving
4. Recovery driving

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
At first I was using one single set of data by running the car. The result is not good, because the loss is huge, and validation data loss is also huge. I then tried to add in flipped images to generalize the data, but the result does not improve. I've realized that it is the problem of the data. I wasn't driving on the center of the road. Therefore, I've decided to generate 2 sets of data carefully to make sure the car stays in the center of the road. These data are from driving forward and backward along the track. After that, I've added recovery data along the track paths, so the model will learn to move back to the center when it is too far away from the center. 

#### 2. Final Model Architecture

I've used the convolutional network introduced by the NVidia self driving team. It is a very powerful network used in training.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Then I use images from left and right side of the camera, modify their angles by a correction of 0.2
![alt text][image3]
![alt text][image4]

I flipped the image and angle for all data, so that will be used to generalize the model to prevent overfitting. I've normalized the data to have a mean of 0, so that it will be trained faster. 

All data were being shuffled and 20% of them are being used as a validation set to check if the model is overfitting.

I've used 20 epochs, because there is a large set of data after preprocessing and data generation. With large dataset, it is unlikely to be overfitting, when training the epochs, the validation success rate is keep dropping indicated that it 
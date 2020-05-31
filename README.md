# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
Deailed writeup:

The final specification that was missing was examples of output images.
I have now included a folder output_images that has 3 samples of left images:
1. Deafult images labeled as left_img_*.jpg

2. Cropped and resized images labeled as cropandresized_img_*.jpg

3. Flipped left images labeled as flipped_img_*.jpg

4. model_architecture.png that shows model architecture

5. project_image.py used for writing these images


Improvements based on last review:
1. My car was going offtrack before. Based on suggestions from a Udacity mentor I put in additional data augmentation to flip just the left and right camera images. This flipping and data augmentation really helped. My car is now able to make sharp turns and never goes offtrack

2. The last reviwer said my model was too simple. But if I a simple model works well, I think the advantage is fewer hyperparameters and it may generalize better. With data augmentation flipping left and right camera images even the simple model works well in navigating sharp turns .

3. But to address last reviwer comments I coded an addition more complex model  

Now, I am including a complex_model.py. This one has two 3x3 convolution layers with max pooling and Relu activation. The depth of fliters is increased from 16 to 32 in the second conv2d layers.

After flattening, I added two densely connected layers with 100 and 20 nuerons and a dropout layer in between. Finally I connect to output dense layer with 1 neuron.

This generates a model_complex.h5 file. 

I then used this on both simple and complex tracks. The vidoe is run2_simpletrack.mp4. This model performs well on both simple and complex tracks, validation loss is low after 5 epochs. I used Adam optimizier.

The results from complex model are good. But the take home is even a simple model with single convolution, max pooling, and final densely connected single neuron performs reasonably well. It has fewer hyperparamaters to train. 

Ultimately, I think the more important step was augmenting data with flipped right and left images that was critical to teach the car to navigate sharp turns.



1. I am including my model.py, drive.py, model.h5 and run1.mp4 files with this report

2. The model.py file was succesful in driving the simulation. I created the video run1.mp4 based on this.

3. I used python generator and fit_generator functions to generate data for training rather than storing data in memory. The genrator function batches the data, and yields training and validation samples in batches of 32 after shuffling. I tried batch sizes of 32, 64 and 128. Only batch size of 32 worked. With batch sizes of 64 and 128 it threw fatal exception and core dumped. Probably Udacity Workspace ran out of memory with larger batch sizes.

4. I cropped 50 pixels from top, 20 from bottom. Then resized image to 32x32x3 using a crop_and_resize function that I defined.

5. Data augmentation: I augmented the provided training data using np.fliplr, and also used right and left camera images. Thus, I augmented training data by a factor of 4. I wanted to explore Crop2D layer in Keras, but documentation said there is no equivalent resize layer in Keras, so I dropped this idea

6. Having a input image size of 32x32x3 with cropping is beneficial size I can then use a simple network to train the data.

7. The simple network I used has a convolution 3x3 layer with depth of 15 followed by Relu activation. Then it has a Maxpooling layer with filter size of 2. Finally it flattens it and connects a densely connected layer with output of 1 to predict steering angle. 

8. This simple network trains fast. With just 1 epocch and batch size of 32, the val_loss was 0.019. Further epochs decreased it to 0.018. Here I am just including 1 epoch of training.

9. With provided training data, the car drives fairly well for long. It ultimately goes off track. I think we need data on how the car adjusts itself when it starts going off track.




Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).


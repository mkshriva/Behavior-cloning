import csv
import cv2
import numpy as np

import datetime
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
lines = []

#load csv file
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#crop resize and change color space of image
def crop_and_resize_change_color_space(image):
    dim = (32, 32)
    #print(image.shape)
    image=np.array(image[80:140,:])
    #print(image.shape)
    image = cv2.cvtColor(cv2.resize(image, dim), cv2.COLOR_BGR2RGB)
    #print(image.shape)
    return image

#generator to yeild processed images for training as well as validation data set
def generator(samples, batch_size = 32):
    num_samples=len(samples)	
    while 1: #Loop forever
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples=samples[offset:offset+batch_size]
            images=[]
            angles=[]
            for batch_sample in batch_samples: #Images become 4 times original length due to data augmentation: flipping center image, left and right camera images
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                print(name)
                center_image = cv2.imread(name)
                center_image=crop_and_resize_change_color_space(center_image)
                center_angle = float(batch_sample[3])
                #Appending original image
                images.append(center_image)
                angles.append(center_angle)
                #Appending flipped image
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)
                # Read in left camera image and steering angle with offset
                left_name = './data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                left_image=crop_and_resize_change_color_space(left_image)
                left_angle=center_angle+0.4
                #Appending left camera image with offset
                images.append(left_image)
                angles.append(left_angle)
                # Read in right camera image and steering angle with offset
                right_name = './data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                right_image=crop_and_resize_change_color_space(right_image)
                right_angle=center_angle-0.3
                #Appending left camera image with offset
                images.append(right_image)
                angles.append(right_angle)
                
            # converting to numpy array
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

         


from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda

#creating model to be trained
model = Sequential()
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(32,32,3) )) #Cropping reduces 50 pixels from top and 20 from bottom
model.add(Conv2D(15, (3, 3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(1))

#compiling and running the model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, \
            steps_per_epoch=len(train_samples)*4, \
            validation_data=validation_generator, \
            validation_steps=len(validation_samples)*4, \
            epochs=1, verbose=1)
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#saving the model
model.save('model.h5')


from keras.models import Sequential
from keras.utils.vis_utils import plot_model
#from keras.models import Model
#import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda

#creating model to be trained
model = Sequential()
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(32,32,3) )) #Cropping reduces 50 pixels from top and 20 from bottom
model.add(Conv2D(16, (3, 3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32, (3, 3) ))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
plot_model(model, to_file='model_architecture.png')


# keras method to print the model summary
model.summary()
import csv
import cv2
import numpy as np
import pandas as pd
import os
import datetime
import random
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
lines = []

#load csv file
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)        #Skip the first header row, else it causes problems later
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

i=0
for sample in train_samples:
    if(i<3):
        name = './data/IMG/'+sample[1].split('/')[-1]
        left_img = cv2.imread(name)
        img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        path='./output_images/left_img_'+ str(i) +'.jpg'
        cv2.imwrite(path,left_img)
        new_img=crop_and_resize_change_color_space(left_img)
        path='./output_images/cropandresized_img_'+ str(i) +'.jpg'
        cv2.imwrite(path,new_img)
        flipped_img=np.fliplr(left_img)
        path='./output_images/flipped_img_'+ str(i) +'.jpg'
        cv2.imwrite(path,flipped_img)
        print(name)
        i+=1
        
    

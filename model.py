
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import json
import random
import csv
import sklearn
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
#from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# Read image paths from csv file
def read_csv():
    # Array to store training data
    training_data = []
    
    with open("./data/driving_log.csv") as csvfile:
        read_file = csv.reader(csvfile)
        for line in read_file:
            training_data.append(line)   
            
    training_data = training_data[1:]

    return training_data

# Split the samples into training and validation samples
def split_samples(training_data):
    # Splitting samples and creating generators - 80 perecnt training data and 20 perecnt validation data
     training_sample, validation_sample = train_test_split(training_data, test_size=0.2)
        
     return training_sample, validation_sample 

# Generate training or validation data with help of python generator
def data_generator(data_samples, batch_size=32):
    # Length of the training or validation sample
    data_sample_length = len(data_samples)

    # Loop forever so the generator never terminates
    while 1: 
        data_samples = sklearn.utils.shuffle(data_samples)
        
        # Loop through every batch
        for offset in range(0, data_sample_length, batch_size):
            # Get the batch data
            batch_data = data_samples[offset : offset + batch_size]	
            print("batch_data:", len(batch_data))
            # arrays to store images and steering angle
            images = []
            steering_angle = []
            
            # arrays to store the augmented images and steering angle
            augmented_images = []
            augmented_steering_angle = []
            
            # arrays to store the total images and steering angle
            total_images = []
            total_angles = []

            # loop through every row in the batch data
            for line in batch_data:
                # From the centre, left and right images, since the car moves in left and right directions
                for col in range(3):
                    # The columns in the csv file which has the path of centre, left and right image
                    source_path = line[col]
                    # Get the last row of the line which is the path of the image
                    file_name = source_path.split('/')[-1]
                    # Get the current path of the image
                    current_path = "./my_data/IMG/" + file_name 
                    # read the image from the path
                    image = cv2.imread(current_path)
                    # Get the image appended
                    images.append(image)
                
                # Now that we have centre, left and right images, we need to add a correction
                # When car turns left, it is steering angle plus correction, and when the car turns right
                # its steering angle - correction
                correction = 0.15
                # Get the steering value from the csv file
                angle = float(line[3])
                # Append the steering measurement in the array for the centre image
                steering_angle.append(angle)
                # Append the steering measurement in the array for the left image
                steering_angle.append(angle + correction)
                # Append the steering measurement in the array for the right image
                steering_angle.append(angle - correction)

            # Flip the images and the steering angle
            for img, ang in zip(images, steering_angle):
                # Flip the image
                flipped_image = cv2.flip(img, 1)
                # Get the augmented image appended
                augmented_images.append(flipped_image) 

                # flip the steering angle
                flipped_steering_angle = float(ang) * -1.0
                # Append the augmented steering angle
                augmented_steering_angle.append(flipped_steering_angle)  
                 
            # append the flipped image
            total_images.extend(images)  
            total_images.extend(augmented_images) 
            total_angles.extend(steering_angle)  
            total_angles.extend(augmented_steering_angle) 

            print("images:", len(images))
            print("steering angle:", len(steering_angle))
            print("augmented image:", len(augmented_images))
            print("augmented steering angle:",len(augmented_steering_angle))
            print("total images:", len(total_images))
            print("total angles:",len(total_angles))            
            
            # Get the X and Y training set: the images are the features per batch
            # and the steering is the label per batch
            X_train_batch, Y_train_batch = np.asarray(total_images), np.asarray(total_angles)

            yield sklearn.utils.shuffle(X_train_batch, Y_train_batch)  
              
# Model to create layers for preprocessing data        
def preprocess_data_layers():
    # Create models layer by layer
    model = Sequential()
    
    # Normalise the model
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=[N, 160, 320, 3], output_shape=[160, 320, 3]))
    # Crop the image- remove the sky and the front part of the car 
    # Crop only from top and bottom, not on left and right
    model.add(Cropping2D(cropping=((70, 25),(0,0))))
    
    return model

# Create nvdia model
def nvdia_cnn(model):
    # Add a convolution layers - 6 filters, 5x5 each
    model.add(Conv2D(24, (5,5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(36, (5,5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(48, (5,5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
#     model.add(Dropout(0.5))
    model.add(Dense(50))
#     model.add(Dropout(0.5))
    model.add(Dense(10))
    # Final is a single node
    model.add(Dense(1))
    
    return model


# Set the batch size
batch_size =  32
# Read csv file
training_data = read_csv()
# Get the combined images of centre, left and right lanes and also the steering angle
X_train, X_valid = split_samples(training_data)

print('Train samples: {}'.format(len(X_train)))
print('Validation samples: {}'.format(len(X_valid)))

# Create generator for training data
train_generator = data_generator(X_train, batch_size=32)
# Create generator for validation data
validation_generator = data_generator(X_valid, batch_size=32)
# Preprocess the data
model = preprocess_data_layers()
# Create nvdia model
model = nvdia_cnn(model)
# Compile the model
model.compile(loss='mse', optimizer='adam')
# Model summary
model.summary()
# Train the model
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(X_train)/batch_size), 
            validation_data=validation_generator, validation_steps=math.ceil(len(X_valid)/batch_size), 
            epochs=3, verbose=1)

model.save('model.h5')
    
    

       
        
        
        
    
    
    


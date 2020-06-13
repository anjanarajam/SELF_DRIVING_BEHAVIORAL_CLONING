
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
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

def read_csv():
    """
    This function reads image paths from csv file

    :param : None
    :return: Training data
    """
    # Array to store training data
    training_data = []
    
    with open("./data/driving_log.csv") as csvfile:
        read_file = csv.reader(csvfile)
        for line in read_file:
            training_data.append(line)  
            
    print(len(training_data))
            
    training_data = training_data[1:]

    return training_data

def split_samples(training_data):
    """
    This function Splits the samples into training and validation samples

    :param training_data: Training data
    :return: Training and validation sample
    """
    # Splitting samples and creating generators - 80 perecnt training data and 20 perecnt validation data
    training_sample, validation_sample = train_test_split(training_data, test_size=0.2)
        
    return training_sample, validation_sample 


def read_batch_data(batch_data):
    """
    This function reads the images and the steering angle batch by batch

    :param batch_data: data batch by batch
    :return: images and steering angle
    """
    # arrays to store images and steering angle
    images = []
    steering_angle = [] 
    
    # loop through every row in the batch data
    for line in batch_data:
        # From the centre, left and right images, since the car moves in left and right directions
        for col in range(3):
            # The columns in the csv file which has the path of centre, left and right image
            source_path = line[col]
            # Get the last row of the line which is the path of the image
            file_name = source_path.split('/')[-1]
            # Get the current path of the image
            current_path = "./data/IMG/" + file_name 
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
    
    return images, steering_angle


def flip_images(images, steering_angle):
    """
    This function flips the images and the steering angle 

    :param images         : images
    :param steering_angle : steering angle
    :return: images and steering angle
    """
    # arrays to store the augmented images and steering angle
    augmented_images = []
    augmented_steering_angle = []  
    
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
    
    return augmented_images, augmented_steering_angle


def data_generator(data_samples, batch_size=32):
    """
    This function generate training or validation data with 
    help of python generator

    :param data_samples : training or vaidation samples
    :param batch_size   : batch size
    :return: Features(X_train) and labels(Y_train)
    """
    
    # Length of the training or validation sample
    data_sample_length = len(data_samples)

    # Loop forever so the generator never terminates
    while 1: 
        # Shuffle the data
        data_samples = sklearn.utils.shuffle(data_samples)        
                
        # Loop through every batch
        for offset in range(0, data_sample_length, batch_size):
            # Get the batch data
            batch_data = data_samples[offset : offset + batch_size]	           
          
            # arrays to store the total images and steering angle
            total_images = []
            total_angles = []
            
            # Get the images and the steering angles batch by batch
            images, steering_angle = read_batch_data(batch_data)
            # Flip the collected images
            augmented_images, augmented_steering_angle = flip_images(images, steering_angle)        
            
            # append the flipped image
            total_images.extend(images)  
            total_images.extend(augmented_images) 
            total_angles.extend(steering_angle)  
            total_angles.extend(augmented_steering_angle)    
            
            # Get the X and Y training set: the total_images are features per batch
            # and the total_angles is the label per batch
            X_train_batch, Y_train_batch = np.asarray(total_images), np.asarray(total_angles)

            yield sklearn.utils.shuffle(X_train_batch, Y_train_batch) 

              
# Model to create layers for preprocessing data        
def preprocess_data_layers():
    """
    This function creates layers for preprocessing the data like
    normalizing and cropping.

    :param : None
    :return: created model
    """
    # Create models layer by layer
    model = Sequential()    
    # Normalise the model
    model.add(Lambda(lambda x: (x / 255.0) - .5, input_shape=(160, 320, 3)))
    # Crop the image- remove the sky and the front part of the car 
    # Crop only from top and bottom, not on left and right
    model.add(Cropping2D(cropping=((70, 25),(0,0))))
    
    return model

def nvdia_cnn(model):
    """
    This function creates the NVDIA model architecture

    :param : created sequential model
    :return: NVDIA model
    """
    # Add a convolution layers - 6 filters, 5x5 each
    model.add(Conv2D(24, (5,5), subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
    model.add(Conv2D(36, (5,5), subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
    model.add(Conv2D(48, (5,5), subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
    model.add(Conv2D(64, (3,3), activation='relu', W_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(Dropout(0.25))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(Dropout(0.25))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(Dropout(0.25))
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
#model.compile(loss='mse', optimizer='adam')
model.compile(optimizer=Adam(lr=1e-4), loss='mse')
# Model summary
model.summary()
# Train the model
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(X_train)/batch_size), 
            validation_data=validation_generator, validation_steps=math.ceil(len(X_valid)/batch_size), 
            epochs=3, verbose=1)
# Save the model
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png')

    
    

       
        
        
        
    
    
    


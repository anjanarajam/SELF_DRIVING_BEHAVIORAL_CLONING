
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import json
import random
import csv
import sklearn


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
#from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# Read image paths from csv file
# Load csv file
def read_csv():
    lines = []
    with open("./data_new/driving_log.csv") as csvfile:
        read_file = csv.reader(csvfile)
        for line in read_file:
            lines.append(line)         

    # skip the first row in csv as it is the header
    lines = lines[1:]
    
    return lines

# Read images and steering meaurements from image paths
def load_images(lines):
    images = []
    steering_angle = []

    for line in lines:
        # From the centre, left and right images, since the car moves in left and right directions
        for col in range(3):
            # The columns in the csv file which has the path of centre, left and right image
            source_path = line[col]
            # Get the last row of the line which is the path of the image
            file_name = source_path.split('/')[-1]
            # Get the current path of the image
            current_path = "./data_new/IMG/" + file_name 
            # read the image from the path
            image = cv2.imread(current_path)
            # Get the image appended
            images.append(image)
        # Now that we have centre, left and right images, we need to add a correction
        # When car turns left, it is steering angle plus correction, abnd when the car turns right
        # its steering angle - correction
        correction = 0.15
        # Get the steering value from the csv file
        angle = float(line[3])
        # Append the steering measurement in the array for the centre image
        steering_angle.append(angle )
        # Append the steering measurement in the array for the left image
        steering_angle.append(angle + correction)
        # Append the steering measurement in the array for the left image
        steering_angle.append(angle - correction)

    print(len(images))
    print(len(steering_angle))

    return images, steering_angle

def data_generator(training_data, batch_size=64):
    num_data = len(training_data)
    # Loop forever so the generator never terminates
    while 1: 
        training_data = sklearn.utils.shuffle(training_data)
        
        for offset in range(0, num_data, batch_size):
            batch_data = training_data[offset : offset + batch_size]
            augmented_images = []
            augmented_steering_angle = []

            # Same as "for image, angle in zip(images, steering_angle):" but in btaches
            for image, angle in batch_data:
                augmented_images.append(image)
                augmented_steering_angle.append(angle)
                flipped_image = cv2.flip(image, 1)
                flipped_steering_angle = float(angle) * -1.0
                augmented_images.append(flipped_image)
                augmented_steering_angle.append(flipped_steering_angle)

        # Get the X and Y training set: the images are the features 
        # and the steering is the label
        X_train, Y_train = np.asarray(augmented_images), np.asarray(augmented_steering_angle)
        
        yield sklearn.utils.shuffle(X_train, Y_train)


# Model to create layers for preprocessing data        
def preprocess_data_layers():
    # Create models layer by layer
    model = Sequential()
    # Normalise the model
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=[160, 320, 3]))
    # Crop the image- remove the sky and the front part of the car 
    # Crop only from top and bottom, not on left and right
    model.add(Cropping2D(cropping=((70, 25),(0,0))))
    
    return model

# Create nvdia model
def nvdia_cnn(model):
    # Add a convolution layers - 6 filters, 5x5 each
    model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(64,3,3, activation='relu'))
    model.add(Conv2D(64,3,3, activation='relu'))
    # model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    # Final is a single node
    model.add(Dense(1))
    
    return model


# Read csv file
lines = read_csv()
# Get the combined images of centre, left and right lanes and also the steering angle
images, steering_angle = load_images(lines)
# Zip the data
training_data = list(zip(images, steering_angle))
# Splitting samples and creating generators - 80 perecnt training data and 20 perecnt validation data
X_train, X_valid = train_test_split(training_data, test_size=0.2)

print('Train samples: {}'.format(len(X_train)))
print('Validation samples: {}'.format(len(X_valid)))

batch_size = 64

# Create generator for training data
train_generator = data_generator(X_train, batch_size=64)
# Create generator for validation data
validation_generator = data_generator(X_valid, batch_size=64)
# Preprocess the data
model = preprocess_data_layers()
# Create nvdia model
model = nvdia_cnn(model)
# Compile the model
model.compile(loss='mse', optimizer='adam')
# Model summary
model.summary()
# Train the model
model.fit_generator(train_generator, steps_per_epoch = len(X_train * 2) // batch_size, validation_data = validation_generator, \
                 nb_val_samples=len(X_valid), nb_epoch=3, verbose=1)

model.save('model.h5')

       
        
        
        
    
    
    


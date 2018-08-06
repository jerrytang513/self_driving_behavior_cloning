import csv 
import cv2
import numpy as np 
import scipy.misc
import PIL 

# Get data from csv file
def get_data(file_path):
    lines = []

    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    center_images = []
    left_images = []
    right_images = []

    center_angles = []
    left_angles = []
    right_angles = []

    throttles = []
    breaks = []
    speeds = []

    car_images = []
    car_angles = []

    correction = 0.2 

    for line in lines[1:]:
        # Center Images
        source_path = line[0]
        image = cv2.imread(source_path)
        center_images.append(image)
        center_images.append(np.fliplr(image))
        angle = float(line[3])
        center_angles.append(angle) 
        center_angles.append(-angle)

        # Left Images 
        source_path = line[1]
        image = cv2.imread(source_path)
        left_images.append(image)
        left_images.append(np.fliplr(image))
        left_angles.append(angle + 0.2) 
        left_angles.append(- angle - 0.2)

        # Right Images
        source_path = line[2]
        image = cv2.imread(source_path)
        right_images.append(image)
        right_images.append(np.fliplr(image))
        right_angles.append(angle - 0.2) 
        right_angles.append(- angle + 0.2)
        

        throttle = float(line[4])
        throttles.append(throttle)

        break_ = float(line[5])
        breaks.append(break_)

        speed = float(line[6])
        speeds.append(speed)
        
    car_images.extend(center_images)
    car_images.extend(left_images)
    car_images.extend(right_images)

    car_angles.extend(center_angles)
    car_angles.extend(left_angles)
    car_angles.extend(right_angles)
    return car_images, car_angles

# cw images 
# ccw images 
# recovery images
# smooth curve images
forward_images, forward_angles =  get_data('data/driving_log.csv')
backward_images, backward_angles = get_data('data/backward.csv')
to_center_images, to_center_angles = get_data('data/backtocenter.csv')
curve_images, curve_angles = get_data('data/curve.csv')

images = []
angles = []

images.extend(forward_images)
images.extend(backward_images)
images.extend(to_center_images)
images.extend(curve_images)

angles.extend(forward_angles)
angles.extend(backward_angles)
angles.extend(to_center_angles)
angles.extend(curve_angles)

X_train = np.array(images)
y_train = np.array(angles)

from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D 
from keras.layers.pooling import MaxPooling2D 

# model 
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20, verbose=1)

model.save('model.h5')

    
    
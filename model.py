import csv 
import cv2
import numpy as np 
import scipy.misc
import PIL 


lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

center_images = []
left_images = []
right_images = []
angles = []
throttles = []
breaks = []
speeds = []

for line in lines[1:]:
    # Center Images
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = filename
    image = cv2.imread(current_path)
    center_images.append(image)
    
    # Left Images 
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = filename 
    image = cv2.imread(current_path)
    left_images.append(image)

    # Right Images
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = filename 
    image = cv2.imread(current_path)
    right_images.append(image)

    angle = float(line[3])
    angles.append(angle) 

    throttle = float(line[4])
    throttles.append(throttle)

    break_ = float(line[5])
    breaks.append(break_)

    speed = float(line[6])
    speeds.append(speed)

X_train = np.array(center_images)
y_train = np.array(angles)

from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D 
from keras.layers.pooling import MaxPooling2D 

#model = Sequential()
#model.add(Conv2D(32, (3, 3), input_shape=(160, 320, 3)))

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')

    
    
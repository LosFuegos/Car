import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def generator(lines, batch_size=32, a=True):
    num_lines = len(lines)
    correction = 0.33
    path = 'DataSet/'
    while 1:
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]
            images = []
            measurements = []

            for batch_line in batch_lines:
                #getting corrected steering measurements
                steering = float(batch_line[3])
                steering_left = steering + correction
                steering_right = steering - correction

                #getting file path for images
                #had some whitespace in my path. Striped those spaces out.
                center = batch_line[0].lstrip()
                left = batch_line[1].lstrip()
                right = batch_line[2].lstrip()

                center_path = path + center
                left_path = path + left
                right_path = path + right

                #adding images to list
                #adding steering measurements to list
                images.append(cv2.imread(center_path, 1))
                measurements.append(steering)
                images.append(cv2.imread(left_path, 1))
                measurements.append(float(steering_left))
                images.append(cv2.imread(right_path, 1))
                measurements.append(float(steering_right))

            #translating images in training set only.
            if a:
                for a in range(0 , len(images)):
                    x, y = translate_image(images[a], measurements[a], 75)
                    images.append(x)
                    measurements.append(y)

            images, measurements = preprocess(images, measurements)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train)

def preprocess(Xtrain, Ytrain):
    images = Xtrain
    measurements = Ytrain
    for i in range(0, len(measurements)):
        if measurements[i] >= -0.4 and measurements[i] <= 0.4 and np.random.uniform(0.1, 1) < 0.70:
            np.delete(images, i, 0)
            np.delete(measurements, i, 0)

    #changing color space to HLS
    for i in range(0, len(images)):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2HLS)
        images[i] = images[i][60:135, 0:320]
        images[i] = cv2.resize(images[i], (200,66))

    #adding flipped images
    for img in range(0, len(images)):
        images.append(np.fliplr(images[img]))
        measurements.append(-measurements[img])
    return images, measurements

#Based on https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ti3uedvq4
def translate_image(image,steering,trans_range):
    #Translation
    #Random Translation
    trans_x = (trans_range * np.random.uniform()) - (trans_range / 2)

    #Getting Steering measurement
    steering_ang = steering + trans_x / trans_range *2 * 0.19

    trans_y = 0

    #warp image
    Trans_M = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    translated_image = cv2.warpAffine(image, Trans_M, (320, 160))

    return translated_image,steering_ang

lines = []
with open("DataSet/driving_logs.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines = shuffle(lines)

print(lines[0:31])
train_lines, validation_lines = train_test_split(lines, test_size=0.25)

from keras.models import  Sequential
from keras.regularizers import l2
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint

train_gen = generator(train_lines, batch_size=32, a=True)
validation_gen = generator(validation_lines, batch_size=32, a=False)

#Based On Nvidia Model.
model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=(66,200,3)))
model.add(Conv2D(24,5,5, subsample=(2,2), border_mode='valid', init='he_normal', activation='elu'))
model.add(Conv2D(36,5,5, subsample=(2,2), border_mode='valid', init='he_normal',  activation='elu'))
model.add(Conv2D(48,5,5, subsample=(2,2), border_mode='valid', init='he_normal', activation='elu'))
model.add(Conv2D(64,3,3, border_mode='valid', init='he_normal',  activation='elu'))
model.add(Conv2D(64,3,3, border_mode='valid', init='he_normal',  activation='elu'))
model.add(Flatten())
model.add(Dense(100, init='he_normal',  activation='elu'))
model.add(Dense(50, init='he_normal',  activation='elu'))
model.add(Dense(10, init='he_normal', activation='elu'))
model.add(Dense(1, init='he_normal'))

#model.load_weights('model_weights.hdf5')

model.compile(loss='mse', optimizer='adam')
checkpoint = ModelCheckpoint('model_weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit_generator(train_gen,samples_per_epoch=25000, validation_data=validation_gen,nb_val_samples=2500,\
                    nb_epoch=10, callbacks=callbacks_list)
model.save_weights('model_weights.h5')
model.save('model.h5')
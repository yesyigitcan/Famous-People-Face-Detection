import cv2 as cv
import numpy as np
import os




train_x = []
train_y = []

test_x = []
test_y = []

artists = ["Brad", "Daniel", "Elon"]

for artist in artists:
    folder_path = '.\\face\\Celebrity\\train\\' + artist
    for image_path in os.listdir(folder_path):
        image = cv.imread(folder_path + "\\" + image_path)
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_gray_resized = cv.resize(image_gray, (150, 150))
        train_x.append(image_gray_resized)
        train_y.append(artist)

    folder_path = '.\\face\\Celebrity\\test\\' + artist
    for image_path in os.listdir(folder_path):
        image = cv.imread(folder_path + "\\" + image_path)
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_gray_resized = cv.resize(image_gray, (150, 150))
        test_x.append(image_gray_resized)
        test_y.append(artist)


from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(
    Conv2D(32, (3, 3), activation='relu')
)
model.add(
    MaxPooling2D(pool_size=(2, 2))
)
model.add(
    Conv2D(32, (3, 3), activation='relu')
)
model.add(
    MaxPooling2D(pool_size=(2, 2))
)
model.add(
    Flatten()
)
model.add(
    Dense(units=150, activation='relu')
)
model.add(
    Dense(units=1, activation='sigmoid')
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=30, batch_size=32)
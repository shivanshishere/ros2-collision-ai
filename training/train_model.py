import tensorflow as tf
import cv2
import numpy as np

# load dataset
# preprocess frames
# build CNN + LSTM model

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# train
model.fit(X_train,y_train,epochs=10)

# save model
model.save("best_collision_model.h5")

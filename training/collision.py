
import pandas as pd

df = pd.read_excel("dataset_database.xlsx")

print(df.head())
print(df.columns)

print("Shape:", df.shape)
print("\n Columns:")
print(df.columns)

print(df.head)
print(df.isnull().sum())

import pandas as pd

df = pd.read_excel("dataset_database.xlsx")

df["label"] = df["collision"].map({"n": 0, "y": 1})

print(df[["subject", "collision", "label"]].head())

df.to_csv("dataset_labels.csv", index=False)
print(" Labels encoded & saved")

SEQ_LEN = 10
FUTURE_OFFSET = 20   # ~2 sec future prediction
IMG_SIZE = 128

df = pd.read_csv("dataset_labels.csv")

future_labels = []
for i in range(len(df) - FUTURE_OFFSET):
    future_labels.append(df["label"].iloc[i + FUTURE_OFFSET])


df = df.iloc[:len(future_labels)]

df["future_collision"] = future_labels

print(df[["subject", "label", "future_collision"]].head(15))

df.to_csv("dataset_future_labels.csv", index=False)
print(" 2-sec future labels created")
print(df["future_collision"].value_counts())

import cv2
import numpy as np
import pandas as pd
import os
import tensorflow as tf

SEQ_LEN = 10
IMG_SIZE = 128

df = pd.read_csv("dataset_future_labels.csv")

def load_sequences(start_idx):
     frames=[]
     for i in range(start_idx, start_idx + SEQ_LEN):
           img_path = os.path.join("dataset", df.loc[i, "subject"])
           img = cv2.imread(img_path)

           if img is None:
            img = np.zeros((IMG_SIZE,IMG_SIZE,3))

           img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
           img = img / 255.0
           frames.append(img)
     return np.array(frames)


X, y = [], []

MAX_SEQ = 3000

for i in range(min(MAX_SEQ, len(df) - SEQ_LEN)):
    X.append(load_sequences(i))
    y.append(df["future_collision"].iloc[i + SEQ_LEN - 1])


X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report


model = Sequential([

    TimeDistributed(
        Conv2D(16,(3,3),activation="relu",padding="same"),
        input_shape=(10,128,128,3)
    ),

    TimeDistributed(MaxPooling2D()),
    TimeDistributed(BatchNormalization()),


    TimeDistributed(Conv2D(32,(3,3),activation="relu",padding="same")),
    TimeDistributed(MaxPooling2D()),
    TimeDistributed(BatchNormalization()),


    TimeDistributed(Conv2D(64,(3,3),activation="relu",padding="same")),
    TimeDistributed(MaxPooling2D()),
    TimeDistributed(BatchNormalization()),


    # Flatten ki jagah ye use karo
    TimeDistributed(GlobalAveragePooling2D()),


    LSTM(32, return_sequences=False),

    Dropout(0.4),

    Dense(32,activation="relu"),

    Dense(1,activation="sigmoid")

])


model.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="binary_crossentropy",
    metrics=["accuracy"]

)

model.summary()


# callbacks

early_stop = EarlyStopping(

    monitor="val_loss",
    patience=6,
    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "best_collision_model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1

)


history = model.fit(

    X,
    y,

    epochs=30,
    batch_size=16,

    validation_split=0.3,

    shuffle=True,

    class_weight={0:1, 1:4},

    callbacks=[early_stop, checkpoint]

)



# prediction

pred = model.predict(X)

# threshold improved
pred = (pred > 0.5).astype(int)


print("Confusion Matrix:")
print(confusion_matrix(y, pred))


print("\nClassification Report:")
print(classification_report(y, pred))

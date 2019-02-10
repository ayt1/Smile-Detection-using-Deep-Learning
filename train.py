# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from MiniVGGNet import MiniVGGNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
import os

data = []
labels = []

with open('labels.txt', 'r') as f:
  for label in f:
    label = label[0]
    labels.append(label)
f.close()

datasetPath = 'data'
label_names = ['no_smile', 'smile']
impaths = sorted(list(paths.list_images(datasetPath)))

SIZE = 32
for impath in impaths:
  image = cv2.imread(impath, 0)
  image = cv2.resize(image, (SIZE,SIZE))
  image = img_to_array(image)
  
  data.append(image)

# convert to numpy array and scale the data
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert labels to vectors
le = LabelEncoder()
labels = np_utils.to_categorical(le.fit_transform(labels), 2)

# split train and test data 
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

model = MiniVGGNet.build(width=SIZE, height=SIZE, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# model checkpoint
checkpoint = ModelCheckpoint('path_to_save_model', monitor="val_loss", save_best_only=True, verbose=1)
callbacks = [checkpoint]

EPOCH = 25
BATCH = 64
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=BATCH, epochs=EPOCH, callbacks=callbacks, verbose=1)

predictions = model.predict(testX, batch_size=BATCH)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

plt.style.use("ggplot")
plt.figure(figsize = (16,12))
plt.plot(np.arange(0, EPOCH), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCH), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCH), H.history["acc"], label="acc")
plt.plot(np.arange(0, EPOCH), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

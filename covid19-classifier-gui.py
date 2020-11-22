
# coding: utf-8

# Author: Yichun Zhao

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import imutils
import cv2
import tkinter as tk
import tkinter.filedialog as fd

# load keras model
print("\n[INFO] LOARDING TRAINED MODEL......\n")
model_loaded = keras.models.load_model(os.path.abspath('')+"/model")
print("\n[INFO] MODEL LOADED! Please select an x-ray image.\n")

root = tk.Tk()
root.withdraw()
file_path = fd.askopenfilename()

#image = cv2.imread(os.path.abspath('')+"/dataset/evaluate/no/NORMAL2-IM-1436-0001.jpeg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (224, 224))
image = image/255
image = image.reshape(1, image.shape[0], image.shape[1], 1)
output = model_loaded.predict(image)
print("\nPROBABILITY:  ", output[0][0])
output = int(round(output[0][0]))
if (output):
    print("\nRESULT:  COVID19 detected.\n")
else:
    print("\nRESULT:  COVID19 not detected.\n")

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 21:41:34 2020

@author: -
"""

import tensorflow as tf
import pathlib
import numpy as np
data_dir = pathlib.Path("D:/Personal/Vtop/Semester 5/Artificial Intellegance/J Comp/raw-img")
model = tf.keras.models.load_model("./Test1")
class_names = ['Butterfly','Cat','Chicken','Cow','Dog','Elephant','Horse','Sheep','Spider','Squirrel']
image = tf.keras.preprocessing.image.load_img(
    "./1.jpg", 
    grayscale=False, color_mode='rgb', target_size=(100, 100),
    interpolation='nearest'
)
img = tf.keras.preprocessing.image.img_to_array(image)
img = tf.expand_dims(img, 0)
prediction_out = model.predict(img)
score = tf.nn.softmax(prediction_out[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

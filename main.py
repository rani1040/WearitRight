import streamlit as st
import os
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
from keras import models
from keras import layers
import numpy as np


your_target_size = (150,150,3)
img = image.load_img("Images_Directory/uploaded_image.jpg",target_size=(your_target_size))

img_array = image.img_to_array(img)
print(img_array.shape)
img_array = np.expand_dims(img_array, axis=0)
print(img_array.shape)
model = load_model('vgg16.h5')
print(model.summary())
prediction = model.predict(img_array)
print(prediction)
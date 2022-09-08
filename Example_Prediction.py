import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
filepath='D:/Plant_leaf_prediction/model.h5'
model=load_model(filepath)
print(model)
plant=cv2.imread('D:/Plant_leaf_prediction/test/test/AppleCedarRust2.JPG')
test_image=cv2.resize(plant,(128,128))
test_image=img_to_array(test_image)/255
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
pred=np.argmax(result,axis=1)


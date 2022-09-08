from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config=ConfigProto()
config.gpu_options_allow_growth=true
session=InteractiveSession(config=config)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
classifier=Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(128,128,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2))
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(Flatten())
classifier.add(Dense(unit=128,activation='relu'))
classifier.add(Dense(unit=10,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
#lay dataset tu may
training_set=train_datagen.flow_from_directory('D:/Plant_leaf_prediction/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
                                               target_size=(128,128),batch_size=50,class_mode='categorical')
valid_set=test_datagen.flow_from_directory('D:/Plant_leaf_prediction/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',
                                           target_size=(128,128),batch_size=50,class_mode='categorical')

labels=training_set.class_indices
print(labels)
classifier.fit_generator(training_set,steps_per_epoch=20,epochs=50,validation_data=valid_set)
classifier_json=classifier.to_json()
with open('model1.json','w') as json_file:
    json_file.write(classifier_json)
    classifier.save_weights('weight_model.h5')
    classifier.save('model.h5')




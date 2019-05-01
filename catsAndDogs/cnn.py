# Convolutional Neural Network


# Part 1 - Building the CNN
# Importing the Keras libraries and packages
import keras
import tensorflow as tf
config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 8}) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os
import math
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution and Pooling
input_size = (128, 128)

classifier.add(Conv2D(32, (3, 3), input_shape=(*input_size, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 is optimal
 
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Step 2 - Flattening
classifier.add(Flatten())

# Step 3 - Full connection
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=1, activation='sigmoid'))

# Step 4 - Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Part 2 - Fitting the CNN to the images


training_set_path ='training_set'
test_set_path = 'test_set'
batch_size = 64

total_number_of_training_pic = 0
total_number_of_test_pic = 0

for dir,subdir,files in os.walk(training_set_path):
    for files in files:
        total_number_of_training_pic +=1
        
for dir,subdir,files in os.walk(test_set_path):
     for files in files:
        total_number_of_test_pic +=1
        
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 class_mode='binary')
 
test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size, 
                                            batch_size=batch_size,
                                            class_mode='binary')

_steps_per_epoch = int(math.ceil(total_number_of_training_pic/batch_size))
_validation_steps = int(math.ceil(total_number_of_test_pic/batch_size))
_wokrers=9
classifier.fit_generator(training_set,
                         steps_per_epoch = _steps_per_epoch,
                         epochs = 100,
                         validation_data=test_set,
                         validation_steps = _validation_steps,
                         max_q_size = 50,
                         workers = _wokrers,
                         shuffle = True,
                         verbose=1)

# Saving the model

def saveModel(pathName):
    classifier.save(pathName)
    print('model saved as'+pathName)    

saveModel('dogAndCatCNNModel.h5')



# Convolutional Neural Network


# Part 1 - Building the CNN


# Importing the Keras libraries and packages
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
input_size = (128, 128)
classifier.add(Conv2D(32, (3, 3), input_shape=(*input_size, 3), activation='relu'))
 
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 is optimal
 
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=1, activation='sigmoid'))

# Step 5 - Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Part 2 - Fitting the CNN to the images


training_set_path ='dataset/training_set'
test_set_path = 'dataset/test_set'
batch_size = 32
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
 
classifier.fit_generator(training_set,
                         steps_per_epoch=33000/batch_size,
                         epochs=100,
                         validation_data=test_set,
                         validation_steps=2000/batch_size,
                         workers=8,
                         max_q_size=100)

# Saving the model

def saveModel(pathName):
    classifier.save(pathName)
    print('model saved as'+pathName)    

saveModel('dogAndCatCNNModel.h5')





import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes=5 #Because we have 5 classes for train and validate

img_rows,img_cols = 48,48
batch_size=8   #how many image you want to give your model to train  at once

train_data=r'C:\Users\Tajwar\Desktop\Machine Learning Project\images\train'
validation_data=r'C:\Users\Tajwar\Desktop\Machine Learning Project\images\validation'

#ImageDataGenerator is used to make multiple images of a same images by different orientation to train

train_datagenerator = ImageDataGenerator(rescale=1./255, #we still need to scale down the images
                                         rotation_range=30, # rotating image
                                         shear_range=0.3, #Image Shearing
                                         zoom_range=0.3, # zooming image by 30%
                                         width_shift_range=0.4, #shift the image according to width
                                         height_shift_range=0.4, #shift the image according to its height
                                         horizontal_flip=True,  #mirroring the image horizontally
                                         vertical_flip=True #mirroring the image vertically
                                         )
#Cross validating the images
validation_datagenerator=ImageDataGenerator(rescale=1./255)

#Modifying the images to train and give to model
train_generator=train_datagenerator.flow_from_directory(train_data,
                                                  color_mode='grayscale', #coverting image to blacknWhite
                                                  target_size=(img_rows,img_cols), #giving image size
                                                  batch_size=batch_size,
                                                  class_mode='categorical', #we have images categories like sad,happy
                                                  shuffle=True #Shuffling the images
                                                  )
validation_generator=validation_datagenerator.flow_from_directory(validation_data,
                                                  color_mode='grayscale',
                                                  target_size=(img_rows,img_cols),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True
                                                  )

model=Sequential()

#Block 1
#Layer 1
model.add(Conv2D(32, #We are using 32 neurons
          (3,3),
          padding='same',
          kernel_initializer='he_normal',
          input_shape=(img_rows,img_cols,1))
          )
model.add(Activation('elu'))
model.add(BatchNormalization())
#Layer 2
model.add(Conv2D(32,
          (3,3),
          padding='same',
          kernel_initializer='he_normal',
          input_shape=(img_rows,img_cols,1))
          )
model.add(Activation('elu'))
model.add(BatchNormalization())
#layer 3 Maxpooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))
#Layer 4 Dropout Layer
model.add(Dropout(0.2))

#Block 2
#Layer 1
model.add(Conv2D(64, #We are using 64 neurons
          (3,3),
          padding='same',
          kernel_initializer='he_normal')
          )
model.add(Activation('elu'))
model.add(BatchNormalization())
#Layer 2
model.add(Conv2D(64,
          (3,3),
          padding='same',
          kernel_initializer='he_normal')
          )
model.add(Activation('elu'))
model.add(BatchNormalization())
#layer 3 Maxpooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))
#Layer 4 Dropout Layer
model.add(Dropout(0.2))

#Block 3
#Layer 1
model.add(Conv2D(128, #We are using 128 neurons
          (3,3),
          padding='same',
          kernel_initializer='he_normal')
          )
model.add(Activation('elu'))
model.add(BatchNormalization())
#Layer 2
model.add(Conv2D(128,
          (3,3),
          padding='same',
          kernel_initializer='he_normal')
          )
model.add(Activation('elu'))
model.add(BatchNormalization())
#layer 3 Maxpooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))
#Layer 4 Dropout Layer
model.add(Dropout(0.2))

#Block 4
#Layer 1
model.add(Conv2D(256, #We are using 256 neurons
          (3,3),
          padding='same',
          kernel_initializer='he_normal')
          )
model.add(Activation('elu'))
model.add(BatchNormalization())
#Layer 2
model.add(Conv2D(256,
          (3,3),
          padding='same',
          kernel_initializer='he_normal')
          )
model.add(Activation('elu'))
model.add(BatchNormalization())
#layer 3 Maxpooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))
#Layer 4 Dropout Layer
model.add(Dropout(0.2))

#Block 5
#Layer 1
#Flattening layer
model.add(Flatten())
model.add(Dense(64,
          kernel_initializer='he_normal')
          )
model.add(Activation('elu'))
model.add(BatchNormalization())
#Layer 2 Dropout Layer
model.add(Dropout(0.5))

#Block 6
#Layer 1
model.add(Dense(64,
          kernel_initializer='he_normal')
          )
model.add(Activation('elu'))
model.add(BatchNormalization())
#Layer 2 Dropout Layer
model.add(Dropout(0.5))

#Block 7 Last Output Layer
#Layer 1
model.add(Dense(num_classes,
          kernel_initializer='he_normal')
          )
model.add(Activation('softmax'))

print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('Emotion_little_vgg.h5',
                             monitor='val_loss', #Validation Loss
                             mode='min',
                             save_best_only=True, #Saves only that model which is best
                             verbose=1
                             )
#if the validation loss is not improving for 3 rounds then stop the training
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3, #stops after 3 round
                          verbose=1,
                          restore_best_weights=True
                          )
#Reducing the learning rate
#if the accuracy is not improving for the 3 rounds then reduce the learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

#Storing it in a callback list
#Callback is used when the model done with epooch it will chose the best epooch with the highest accuracy
callbacks = [earlystop,checkpoint,reduce_lr]

#Compiling the Model
model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001), #Using Adam Optimizer
              metrics=['accuracy'])

nb_train_samples = 24176
nb_validation_samples = 3006
#model will go 25 times over all images in the folders
epochs=25

#For Starting the training
history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)

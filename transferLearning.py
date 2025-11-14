#Loading the packages
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt 

IMAGE_SIZE = [224,224]
train_path = 'mydata/training_set'
test_path = 'mydata/test_set'

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet',include_top=False)

#Don't train existing layers
for layer in vgg.layers:
    layer.trainable = False

#Add new top layer
x = Flatten()(vgg.output)
prediction = Dense(2,activation='softmax')(x)

#Create the model
model = Model(inputs=vgg.input, outputs=prediction)
#print(model.summary())

#Build the model
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy'] 
)

#Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    train_path,
    target_size = (224,224),
    batch_size=16,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_path,
    target_size = (224,224),
    batch_size=16,
    class_mode='categorical'
)

classifier = model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch = 368 // 16,
    validation_steps = 172 // 16 
)

#Saving the model
import h5py
model.save('vgg16_model.h5')

#Performance
plt.plot(classifier.history['accuracy'])
plt.plot(classifier.history['val_accuracy'])
plt.title('Accuracy of CNN')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

plt.plot(classifier.history['loss'])
plt.plot(classifier.history['val_loss'])
plt.title('Loss of CNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
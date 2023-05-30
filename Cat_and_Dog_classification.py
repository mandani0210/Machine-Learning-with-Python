import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential


from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


os.environ['KAGGLE_USERNAME'] = 'enter your username'
os.environ['KAGGLE_KEY'] = 'enter your key'


!kaggle datasets download -d chetankv/dogs-cats-images


with zipfile.ZipFile('dogs-cats-images.zip', 'r') as zip_ref:
    zip_ref.extractall('cat_dog_dataset')
    
    
base_dir = 'cat_dog_dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')


# Normalize pixel values to the range [0, 1]
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)


# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    '/content/cat_dog_dataset/dataset/training_set',
    target_size=(150, 150),  # Specify the target size of the images
    batch_size=32,
    class_mode='binary'
)


validation_generator = validation_datagen.flow_from_directory(
    '/content/cat_dog_dataset/dataset/test_set',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    # steps_per_epoch=800,
    epochs=3,
    validation_data=validation_generator,
    # validation_steps=800
)

test_loss, test_acc = model.evaluate(validation_generator, steps=25)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


prediction = model.predict(validation_generator)
if (prediction[0][0] >= 0.5):
    predicted_class = 'dog'
    probability = prediction[0][0]
else:
    predicted_class = 'cat'
    probability = 1 - prediction[0][0]

print(f"Predicted class: {predicted_class}")
print(f"Probability: {probability}")












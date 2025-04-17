import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Set paths
data_dir = '../data/processed_images'
labels_file = '../data/labels.csv'
model_save_path = '../models/face_recognition_model.h5'

# Load labels
labels = pd.read_csv(labels_file)
num_classes = labels['label'].nunique()

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create a model
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Train the model
def train_model():
    input_shape = (100, 100, 3)  # Adjust based on your image size
    model = create_model(input_shape)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Prepare data
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(100, 100),
        batch_size=32,
        class_mode='sparse'
    )

    # Checkpoint to save the best model
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)

    # Fit the model
    model.fit(train_generator, epochs=50, callbacks=[checkpoint])

if __name__ == "__main__":
    train_model()
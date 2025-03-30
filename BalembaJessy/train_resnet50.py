import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43  # Adjust this to match your dataset

# Load dataset (replace with your own data loading function)
def load_data(data_dir):
    images = []
    labels = []
    
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(category)

    images = np.array(images) / 255.0  # Normalize
    labels = np.array(labels)
    
    return images, labels

# Load your train and test datasets
X_train, y_train = load_data("path/to/train_data")
X_test, y_test = load_data("path/to/test_data")

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CATEGORIES)

# Load ResNet50 model pre-trained on ImageNet, without the top layer
base_model = ResNet50(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze the base model

# Build the model on top of ResNet50
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CATEGORIES, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the trained model
model.save("resnet50_model.h5")
print("ResNet50 model saved as resnet50_model.h5")

import cv2
import numpy as np
import tensorflow as tf
import sys

IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43

def load_and_predict(image_path, model):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)  # Add batch dimension

    # Predict
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return predicted_class, confidence

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python test_model.py image_path")

    image_path = sys.argv[1]

    # Load the trained model
    model_path = "C:/Users/Balemba/Desktop/AIProject/BalembaJessy/best_model.h5"
    model = tf.keras.models.load_model(model_path)

    # Predict
    predicted_class, confidence = load_and_predict(image_path, model)
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()

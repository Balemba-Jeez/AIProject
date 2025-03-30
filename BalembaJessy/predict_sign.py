import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

IMG_WIDTH = 30
IMG_HEIGHT = 30

# Load the trained model
model_path = "C:/Users/Balemba/Desktop/AIProject/BalembaJessy/traffic_model.h5"
model = tf.keras.models.load_model(model_path)


def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)  # Add batch dimension

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return predicted_class, confidence

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        predicted_class, confidence = predict_image(file_path)
        result_label.config(text=f"Predicted: {predicted_class}, Confidence: {confidence:.2f}%")

        img = Image.open(file_path)
        img = img.resize((200, 200))  # Resize for display
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

# Create GUI
root = tk.Tk()
root.title("Traffic Sign Classifier")

frame = tk.Frame(root)
frame.pack(pady=20)

btn = tk.Button(frame, text="Upload Image", command=upload_image)
btn.pack()

img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root, text="Upload an image to classify", font=("Arial", 12))
result_label.pack()

root.mainloop()

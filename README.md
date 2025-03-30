# Traffic Sign Recognition Project

## Overview
This project uses a neural network to classify traffic signs. The model is trained on a traffic sign dataset and predicts the type of traffic sign from a given image. A simple Tkinter GUI is provided to upload an image and see the prediction.

## Folder Structure

- **traffic.py**: Python script to train the model.
- **best_model.h5**: The saved best model after training.
- **predict_sign.py**: Python script with a Tkinter GUI for image upload and prediction.
- **images/**: Folder containing 10 sample traffic sign images (used for testing).
- **other_models/** (optional): Folder for any additional models you trained.

## Installation and Setup

1. **Install dependencies**:
    - Install TensorFlow:
      ```bash
      pip install tensorflow
      ```
    - Install Tkinter (if not installed already):
      ```bash
      sudo apt-get install python3-tk
      ```

2. **Clone the repository**:
    - Clone the repo to your local machine:
      ```bash
      git clone https://github.com/Balemba-Jeez/AIProject.git
      ```

## Training the Model

1. In the `traffic.py` file, the neural network is trained on the traffic sign dataset.
2. Run `traffic.py` to train the model and save the trained model as `best_model.h5`:
   ```bash
   python traffic.py

# UrbanSoundClassifier

**UrbanSoundClassifier** is a deep learning project focused on classifying environmental sounds using a convolutional neural network (CNN). It leverages the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html), which contains thousands of audio clips categorized into 10 sound classes such as dog bark, gun shot, engine idling, and more.

## 🎯 Objective

The goal of this project is to:
- Preprocess environmental audio data for machine learning.
- Extract meaningful features using Mel-frequency cepstral coefficients (MFCCs).
- Train a CNN to classify short audio clips into 10 urban sound categories.
- Evaluate the model’s performance and save the best model.

## 📓 Notebook Overview

The notebook `Audio_Classification.ipynb` includes the following steps:

1. **Exploratory Data Analysis**  
   - Visualizes sample audio signals and their waveforms.
   - Displays class distribution in the dataset.

2. **Data Preprocessing**  
   - Loads the audio files and extracts MFCC features.
   - Encodes class labels into integers.
   - Splits data into training and testing sets.

3. **Model Architecture**  
   - A simple CNN built using Keras.
   - Input: 2D MFCC feature matrix.
   - Layers include Conv2D, MaxPooling, Flatten, Dense, and Dropout.

4. **Model Training and Evaluation**  
   - Uses categorical cross-entropy loss and Adam optimizer.
   - Evaluates accuracy on the test set.
   - Saves the trained model to `saved_models/`.

5. **Result Analysis**  
   - Shows classification report and confusion matrix.
   - Discusses strengths and potential improvements.

## 📁 Folder Contents

- `Audio_Classification.ipynb` – The main notebook.
- `saved_models/` – Contains the saved Keras model.
- `data/` – Includes a text file with instructions for downloading the UrbanSound8K dataset.

## 🛠 Tools & Libraries

- Python, Keras, TensorFlow
- librosa, NumPy, pandas
- scikit-learn, matplotlib, seaborn

## 🌐 Dataset

The UrbanSound8K dataset is not included due to size.  
Instructions for downloading are provided in `data/README.txt`.

## ✨ Highlights

- Practical application of CNNs on audio classification.
- Feature engineering with MFCCs.
- Real-world dataset handling with imbalanced classes.

---

> This repository is intended as an educational resource demonstrating how to preprocess, model, and evaluate audio data for classification using deep learning.

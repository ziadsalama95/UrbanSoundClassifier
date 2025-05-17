# UrbanSoundClassifier

This repository contains a Jupyter notebook demonstrating environmental sound classification using a neural network trained on the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html).

## Overview

The notebook `Audio_Classification.ipynb`:
- Loads the UrbanSound8K dataset and performs exploratory data analysis (EDA).
- Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio files.
- Trains a neural network to classify sounds into 10 categories.
- Evaluates the model and predicts classes for new audio files.

## How to Use

1. Clone this repository.
2. Download the UrbanSound8K dataset and place it in the `data/` directory (adjust paths in the notebook if needed).
3. Install required libraries: `pandas`, `numpy`, `librosa`, `keras`, `matplotlib`, `IPython`.
4. Run the notebook cells sequentially to train the model and test predictions.

## Results

The model achieves a validation accuracy of approximately 74.81% after 52 epochs (see notebook for details).
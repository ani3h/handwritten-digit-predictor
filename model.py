import random
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import KFold
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('Train data size:', train_images.shape, train_labels.shape)
print('Test data size:', test_images.shape, test_labels.shape)

# Function to apply thresholding


def apply_thresholding(x):
    thresholded_images = np.zeros_like(x)
    for i in range(x.shape[0]):
        image_data = x[i]
        image_data = image_data.reshape(28, 28, 1).astype(np.uint8)
        _, image_data = cv2.threshold(image_data, 120, 255, cv2.THRESH_BINARY)
        thresholded_images[i] = image_data.reshape(28, 28, 1)
    return thresholded_images


# Prepare training and testing data
X_train = train_images.reshape(-1, 28, 28, 1)  # Reshape for CNN
y_train = train_labels
X_train_th = apply_thresholding(X_train)
X_train_th = X_train_th.astype("float32") / 255.0
y_train = to_categorical(y_train)

# Prepare test data
X_test = test_images.reshape(-1, 28, 28, 1)  # Reshape for CNN
X_test_th = apply_thresholding(X_test)
X_test_th = X_test_th.astype("float32") / 255.0

# Function to create the CNN model


def create_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Define KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
foldNo = 1
accuracyPerFold = []

# K-Fold Cross-Validation
for trainIndex, valIndex in kf.split(X_train_th):
    print(f'Fold {foldNo}')
    print(f'Shape of train index: {trainIndex.shape}')
    print(f'Shape of val index: {valIndex.shape}')

    X_train_fold, X_val_fold = X_train_th[trainIndex], X_train_th[valIndex]
    y_train_fold, y_val_fold = y_train[trainIndex], y_train[valIndex]

    model = create_cnn_model()

    modelHistory = model.fit(X_train_fold, y_train_fold,
                             epochs=10, validation_data=(X_val_fold, y_val_fold))

    scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    accuracyPerFold.append(scores[1] * 100)
    print(f'Fold {foldNo} - Validation Accuracy: {scores[1] * 100:.2f}%')

    foldNo += 1

print(f'Validation Accuracy for each fold: {accuracyPerFold}')
print(f'Average Validation Accuracy: {np.mean(accuracyPerFold):.2f}%')

# Prediction on test set
prediction = model.predict(X_test_th)
result = np.argmax(prediction, axis=1)
result = pd.Series(result, name="Label")

# Create submission DataFrame (this part is optional, depending on how you want to save your results)
submission = pd.concat(
    [pd.Series(range(1, len(result) + 1), name="ImageId"), result], axis=1)
submission.sample(10)

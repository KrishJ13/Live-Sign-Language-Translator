import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import time

"""
Preparing the dataset.
PROBLEM: Dataset downloaded from Kaggle onto system is simply too large and uses all of system's memory during training
process.

SOLUTION: Statistical Sampling
Taking random samples of images per class, lets say 50 images per class each time, we can hope to 
train our neural network

STEPS:
- Randomly Sample Data for Each Batch: For each class, you'll need to randomly pick 20 images and then combine these into a batch.
- Shuffle the Batch - once the images are gathered, shuffle the entire batch before feeding it to the neural network.
- One-hot Encode the Labels: Create one-hot encoded labels for the 29 classes.
"""

# Constants
dataset_dir = "C:\\Users\\krish\\Downloads\\ASL_Alphabet_Dataset\\asl_alphabet_train"
num_classes = 29
num_images_per_class = 10
img_size = (180, 180)  # Resized image size
batch_size = num_images_per_class * num_classes  # 580 images per batch
epochs = 50  # Number of epochs to train the model
batch_number = 0

# Function for one-hot encoding
def one_hot_encode(label_index, num_classes):
    one_hot = [0] * num_classes
    one_hot[label_index] = 1
    return one_hot


# Function to load a batch of data
def load_batch(dataset_dir):
    batch_images = []
    batch_labels = []

    # Create a mapping for class names to integer labels
    class_to_label = {class_name: idx for idx, class_name in enumerate(sorted(os.listdir(dataset_dir)))}

    for class_folder in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_folder)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

            # Randomly select 20 images per class
            selected_images = random.sample(images, min(num_images_per_class, len(images)))

            for image in selected_images:
                img_path = os.path.join(class_path, image)

                # Load image and preprocess
                img = Image.open(img_path).resize(img_size)
                img = np.array(img) / 255.0  # Normalize image to [0,1]
                batch_images.append(img)

                # Get the class label index and one-hot encode it
                label_index = class_to_label[class_folder]
                one_hot_label = one_hot_encode(label_index, num_classes)
                batch_labels.append(one_hot_label)

    # Convert to numpy arrays
    batch_images = np.array(batch_images)
    batch_labels = np.array(batch_labels)

    # Shuffle the batch
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    batch_images = batch_images[indices]
    batch_labels = batch_labels[indices]
    return batch_images, batch_labels


# Define the Convolutional Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(180, 180, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    # Start the timer
    start_time = time.time()

    # Load a batch for this epoch
    batch_images, batch_labels = load_batch(dataset_dir)

    # Measure the time taken for loading the batch
    elapsed_time = time.time() - start_time
    print(f"Time to load batch: {elapsed_time:.2f} seconds")

    # Train on the current batch
    model.fit(batch_images, batch_labels, epochs=1, batch_size=batch_size)




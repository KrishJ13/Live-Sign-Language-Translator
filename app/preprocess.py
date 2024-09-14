import cv2  # OpenCV library for image processing

"""
This script contains utility functions for handling and processing hand images, so that they fit the structure of the training data
and can thus be inputted into the CNN model
"""

# Function to crop the hand image from the frame based on the given bounding box coordinates
def hand_image(x_min, y_min, x_max, y_max, frame):
    return frame[y_min:y_max, x_min:x_max]  # Return the cropped section of the image

# Function to resize the hand image to a fixed size (200x200 pixels) w
def fix_size(image):
    return cv2.resize(image, (200, 200))


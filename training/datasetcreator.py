import os
import random
import shutil

"""
The initial dataset from Kaggle is 5GB. Each letter (a class) contains over 3000 images, and loading all of this
into memory can lead to 2 things: 
- Longer training times of the CNN
- All of system memory being used during the training process
"""

"""
The solution is to take a subset of the larger dataset. By randomly selecting 500 images from each class folder, 
we can reduce the size of the dataset and increase training times.
"""

# Target dataset folder
root_dataset_dir = "C:\\Users\\krish\\Downloads\\ASL_Alphabet_Dataset\\asl_alphabet_train"
new_dataset_dir = 'C:\\Users\\krish\\Downloads\\ASL_training_Data_Lite'

# Number of images to select from each class
num_images_per_class = 500

# Ensure the new dataset directory exists
if not os.path.exists(new_dataset_dir):
    os.makedirs(new_dataset_dir)

# Loop through each class folder in the root dataset
for class_folder in os.listdir(root_dataset_dir):
    class_path = os.path.join(root_dataset_dir, class_folder)

    if os.path.isdir(class_path):
        # Get a list of all images in the class folder
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Randomly select 500 images (or fewer if there aren't that many)
        selected_images = random.sample(images, min(num_images_per_class, len(images)))

        # Create a corresponding folder in the new dataset
        new_class_path = os.path.join(new_dataset_dir, class_folder)
        if not os.path.exists(new_class_path):
            os.makedirs(new_class_path)

        # Copy selected images to the new dataset folder
        for image in selected_images:
            src_path = os.path.join(class_path, image)
            dest_path = os.path.join(new_class_path, image)
            shutil.copy(src_path, dest_path)

        print(f"Copied {len(selected_images)} images from {class_folder} to new dataset.")

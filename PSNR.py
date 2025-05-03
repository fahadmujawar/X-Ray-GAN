import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

# Function to load images from directories
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        if img is not None:
            images.append(img)
    return images

# Directories containing original and generated images
original_directory = 'D:/CAPSTONE/CODE/PROGAN/Machine-Learning-Collection-master/ML/Pytorch/GANs/ProGAN/PSNR/COVIDOG'
generated_directory = 'D:/CAPSTONE/CODE/PROGAN/Machine-Learning-Collection-master/ML/Pytorch/GANs/ProGAN/PSNR/COVIDGEN'

# Load images from directories
original_images = load_images_from_directory(original_directory)
generated_images = load_images_from_directory(generated_directory)

# Check if the number of images in both directories is the same
if len(original_images) != len(generated_images):
    print("Number of images in the directories does not match.")
    exit()

# Calculate PSNR for each pair of corresponding images
psnr_scores = []
for i in range(len(original_images)):
    psnr = peak_signal_noise_ratio(original_images[i], generated_images[i])
    psnr_scores.append(psnr)

# Calculate average PSNR score
average_psnr = np.mean(psnr_scores)
print(f"Average PSNR: {average_psnr}")

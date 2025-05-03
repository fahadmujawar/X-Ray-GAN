#IMAGE PRE-PROCESSING

import os
import numpy as np
from PIL import Image
from PIL import ImageFilter

def convert_to_grayscale(image):
    return image.convert('L')

def resize_image(image, new_size):
    return image.resize(new_size)

#Sharpening image
def sharpen_image(image):
    # Apply unsharp mask filter to sharpen the image
    sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    return sharpened_image

def normalize_image(image):
    image_array = np.array(image)
    arr_min = image_array.min()
    arr_max = image_array.max()
    arr_range = arr_max - arr_min
    scaled = (image_array - arr_min) / float(arr_range)
    normalized = -1 + (scaled * 2)
    return normalized

# def histogram_equalization2(image):
#     img_array = np.asarray(image)
#     histogram_array = np.bincount(img_array.flatten(), minlength=256)
#     num_pixels = np.sum(histogram_array)
#     histogram_array = histogram_array / num_pixels
#     chistogram_array = np.cumsum(histogram_array)
#     transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
#     img_list = list(img_array.flatten())
#     eq_img_list = [transform_map[p] for p in img_list]
#     eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)
#     eq_img = Image.fromarray(eq_img_array, mode='L')
#     return eq_img

def histogram_equalization(image):
    img_array = np.asarray(image)

    # Convert the image data to integer range [0, 255]
    img_array_int = (img_array * 127.5 + 127.5).astype(np.uint8)

    histogram_array = np.bincount(img_array_int.flatten(), minlength=256)
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / num_pixels
    chistogram_array = np.cumsum(histogram_array)
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
    img_list = list(img_array_int.flatten())
    eq_img_list = [transform_map[p] for p in img_list]
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)
    eq_img = Image.fromarray(eq_img_array, mode='L')
    return eq_img

def process_and_save_image(input_image_path, output_image_path):
    image = Image.open(input_image_path)
    image = convert_to_grayscale(image)
    image = resize_image(image, (256, 256))
    image = normalize_image(image)
    image = histogram_equalization(image)
    image.save(output_image_path)
    image.close()

# Define the base directory containing your images
base_directory = r'C:\kuval\PES CSE\CAPSTONE\DATASET\Viral Pneumonia'

# Define the directory to save preprocessed images
output_directory = r'C:\kuval\PES CSE\CAPSTONE\DATASET\preprocess_VIRAL-PNEUMONIA'
os.makedirs(output_directory, exist_ok=True)

# Process and save preprocessed images
for i, path in enumerate(os.listdir(base_directory)):
    file_path = os.path.join(base_directory, path)
    output_path = os.path.join(output_directory, f'preprocessed_{i}.jpg')
    process_and_save_image(file_path, output_path)

print("Preprocessing and saving complete.")

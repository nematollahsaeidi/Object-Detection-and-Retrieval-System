import cv2
import numpy as np
import time

from skimage import color
from skimage.feature import local_binary_pattern, hog
from scipy.spatial.distance import cdist


def lbp_histogram(image):
    patterns = local_binary_pattern(image, 8, 1)
    # patterns = hog(img)
    hist, _ = np.histogram(patterns, bins=np.arange(2 ** 8 + 1), density=True)
    return hist


def lbp_histogram_rgb(image_input, location_image):
    directory_image = location_image + image_input
    image_input = cv2.imread(directory_image)
    image_input = color.rgb2gray(image_input)
    patterns = local_binary_pattern(image_input, 8, 1)
    # patterns = hog(img)
    hist, _ = np.histogram(patterns, bins=np.arange(2 ** 8 + 1), density=True)
    return hist

def rank_images_texture(image_input, images, images_label):
    texture_dominant_time = time.time()
    dominant_texture_image = lbp_histogram(image_input)
    dominant_texture_image = dominant_texture_image.reshape((1, -1))
    print(f"--- texture dominant takes {(time.time() - texture_dominant_time):.3f} seconds  ---")

    dominant_texture_images = [lbp_histogram(i) for i in images]

    distance_texture_time = time.time()
    closest_texture = 1 - cdist(dominant_texture_image, dominant_texture_images, metric='cosine')
    print(f"--- texture distance takes {(time.time() - distance_texture_time):.3f} seconds  ---")

    closest_texture = closest_texture.ravel()
    sorted_index = np.argsort(-closest_texture, axis=0)
    closest_texture_sort = [images_label[i] for i in sorted_index]
    print(sorted_index, closest_texture_sort)
    return sorted_index, closest_texture_sort

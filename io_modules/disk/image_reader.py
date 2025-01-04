import cv2
from skimage import color


def get_gray_image(image_path):
    # image = cv2.imread(location_image + image)
    image = cv2.imread(image_path)
    image = color.rgb2gray(image)
    return image

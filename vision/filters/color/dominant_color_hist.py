import math
import time
import numpy
from PIL import Image

from vision.filters.color.color_distance import distance_color
from vision.filters.color.dominant_color_algorithms.algolia.color_extractor.image_to_color import ImageToColor


def dominant_color_histogram(image_input, location_image):
    width, height = 500, 500
    image_input = Image.open(location_image + image_input)
    image_input = image_input.resize((width, height))

    pixels = image_input.getcolors(width * height)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])

    dominant_color_sort = sorted_pixels[-1][1]
    # print(f'dominant_color: {dominant_color_sort}')
    return dominant_color_sort


def rank_images_color(image_input, images, images_label, location_image, location_images):
    # clustering_dominant_color = FindDominantColors()
    image_to_color = ImageToColor()

    color_dominant_time = time.time()
    # dominant_color = dominant_color_histogram(image_input, location_image)
    # dominant_color = clustering_dominant_color.dominant_colors_clustering(location_image + image_input, 3)[0][0]
    color_name, dominant_color = image_to_color.get_dominant_color_by_dir(location_image + image_input)
    print(f"--- color dominant takes {(time.time() - color_dominant_time):.3f} seconds  ---")

    # colors_images = [dominant_color_histogram(i, location_images) for i in images]

    # colors_images = [clustering_dominant_color.dominant_colors_clustering(location_images + i, 3)[0][0] for i in images]
    colors_images = [image_to_color.get_dominant_color_by_dir(location_images + i)[1] for i in images]

    distance_color_time = time.time()
    # closest_colors = sorted(colors_images, key=lambda color: distance_color(color, dominant_color))
    closest_colors = [distance_color(i, dominant_color) for i in colors_images]
    print(f"--- color distance takes {(time.time() - distance_color_time):.3f} seconds  ---")

    top = numpy.argsort(closest_colors, axis=0)
    closest_colors_sort = [images_label[i] for i in top]
    print(top, closest_colors_sort)
    return top, closest_colors_sort

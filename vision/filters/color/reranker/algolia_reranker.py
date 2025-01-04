import numpy as np

from vision.filters.color.color_distance import distance_color
from vision.filters.color.dominant_color_algorithms.algolia.color_extractor.image_to_color import ImageToColor


class AlgoliaReranker:
    def __init__(self):
        self.image_to_color = ImageToColor()

    def rank(self, img, images_rgb, images_index):
        dominant_color = self.image_to_color.get_dominant_color_rgb(img)
        print(self.image_to_color.get_dominant_color_text(img))
        closest_colors = [distance_color(i, dominant_color) for i in images_rgb]
        sorted_index = np.argsort(closest_colors, axis=0)
        sorted_by_color = [images_index[i] for i in sorted_index]
        return sorted_by_color

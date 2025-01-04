import time
import cv2
import webcolors
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sys import exit
from PIL import Image


def make_histogram(cluster):
    n_labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    hist, _ = np.histogram(cluster.labels_, bins=n_labels)
    hist = hist.astype('float32')
    hist /= hist.sum()
    return hist


def make_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    HSV_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    hue, sat, val = HSV_bar[0][0]
    return bar, (red, green, blue), (hue, sat, val)


def sort_colors(HSV_list):
    bars_with_indexes = []
    for index, HSV_val in enumerate(HSV_list):
        bars_with_indexes.append((index, HSV_val[0], HSV_val[1], HSV_val[2]))
    bars_with_indexes.sort(key=lambda elem: (elem[1], elem[2], elem[3]))
    return [item[0] for item in bars_with_indexes]


def closest_color(request_color):
    min_color = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_RGB, g_RGB, b_RGB = webcolors.hex_to_rgb(key)
        min_r_RGB = (r_RGB - request_color[0]) ** 2
        min_g_RGB = (g_RGB - request_color[1]) ** 2
        min_b_RGB = (b_RGB - request_color[2]) ** 2
        min_color[(min_r_RGB + min_g_RGB + min_b_RGB)] = name
    return min_color[min(min_color.keys())]


def get_color_name(request_color):
    try:
        name_closest = name_actual = webcolors.rgb_to_name(request_color)
    except ValueError:
        name_closest = closest_color(request_color)
        name_actual = None
    return name_actual, name_closest


def show_bars(values_RGB_HSV_bars):
    sorted_bar_indexes = sort_colors(values_RGB_HSV_bars['HSV_values'])
    sorted_bars = [values_RGB_HSV_bars['bars'][idx] for idx in sorted_bar_indexes]
    cv2.imshow('Sorted by HSV values:', np.hstack(sorted_bars))
    cv2.waitKey(0)


def get_HSV_RGB_sort(values_df, sorted_bar_indexes):
    HSV_sort = [values_df.iloc[j]['HSV_values'] for j in sorted_bar_indexes]
    RGB_sort = [values_df.iloc[j]['RGB_values'] for j in sorted_bar_indexes]
    return HSV_sort, RGB_sort


class FindDominantColors:
    def __init__(self):
        self.values_RGB_HSV_bars = dict()
        self.values_RGB_HSV_bars['RGB_values'] = []
        self.values_RGB_HSV_bars['HSV_values'] = []
        self.values_RGB_HSV_bars['bars'] = []
        self.width, self.height = 50, 50

    def __read_image(self, image):
        if image.endswith(".jpg") or image.endswith(".png") or image.endswith(".jpeg") or image.endswith(".jfif") or image.endswith(".webp"):
            # TODO: thumbnail instead of resize
            image = cv2.resize(cv2.imread(image), (self.width, self.height))
            height, width, _ = np.shape(image)
            image = image.reshape((height * width, 3))
            return image
        else:
            print(f'please select an image with format jpg, png ,jpeg or jfif')
            exit(-1)

    def dominant_colors_clustering(self, image_path, n_clusters=7):
        image = self.__read_image(image_path)
        clustering_time = time.time()
        clusters_fit = KMeans(n_clusters=n_clusters, init='k-means++').fit(image)
        print(f"--- clustering takes {(time.time() - clustering_time):.3f} seconds  ---")

        histogram = make_histogram(clusters_fit)
        hist_percentage_sort = np.sort(histogram)
        combined = zip(histogram, clusters_fit.cluster_centers_)
        combined = sorted(combined, key=lambda x: x[0], reverse=True)
        for index, rows in enumerate(combined):
            bar, RGB, HSV = make_bar(100, 100, rows[1])
            print(f'Bar {index + 1},  RGB values: {RGB},  HSV values: {HSV}')
            self.values_RGB_HSV_bars['RGB_values'].append(RGB)
            self.values_RGB_HSV_bars['HSV_values'].append(HSV)
            self.values_RGB_HSV_bars['bars'].append(bar)

        sorted_bar_indexes = sort_colors(self.values_RGB_HSV_bars['HSV_values'])
        values_df = pd.DataFrame(self.values_RGB_HSV_bars, columns=['RGB_values', 'HSV_values', 'bars'])
        color_names = [get_color_name(values_df.iloc[j]['RGB_values']) for j in sorted_bar_indexes]

        HSV_sort, RGB_sort = get_HSV_RGB_sort(values_df, sorted_bar_indexes)
        print(f'values_HSV_sort: {HSV_sort}\nvalues_RGB_sort: {RGB_sort}\nhist_sort_percentage: '
              f'{hist_percentage_sort}\ncolor_names: {color_names}')
        return RGB_sort, hist_percentage_sort, color_names

    def dominant_color_histogram(self, image):
        image = Image.open(image)
        image = image.resize((self.width, self.height), resample=0)

        pixels = image.getcolors(self.width * self.height)
        sorted_pixels = sorted(pixels, key=lambda t: t[0])

        dominant_color = sorted_pixels[-1][1]
        print(f'dominant_color: {dominant_color}')
        return dominant_color


# img = 'im (9).jpg'
# cp = FindDominantColors()
# all_time = time.time()
# values_RGB_sort, color_name, hist_sort_percentage = cp.dominant_colors_clustering(img)
# print(f"--- dominant_colors_clustering takes {(time.time() - all_time):.3f} seconds  ---")


# img = 'im (9).jpg'
# cp = FindDominantColors()
# all_time = time.time()
# dominant_color_hist = cp.dominant_color_histogram(img)
# print(f"--- dominant_color_histogram takes {(time.time() - all_time):.3f} seconds  ---")
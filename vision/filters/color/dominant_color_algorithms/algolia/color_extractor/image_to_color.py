import numpy as np
import skimage.io

from .back import Back
from .cluster import Cluster
from .name import Name
from .resize import Resize
from .selector import Selector
from .skin import Skin
from .task import Task


class ImageToColor(Task):
    def __init__(self, samples=None, labels=None, settings=None):

        self.color_names = './models/dominant_color/color_names.npz'
        npz = np.load(self.color_names)
        settings = {'debug': None}
        samples, labels = npz['samples'], npz['labels']

        if settings is None:
            settings = {}

        super(ImageToColor, self).__init__(settings)
        self._resize = Resize(self._settings['resize'])
        self._back = Back(self._settings['back'])
        self._skin = Skin(self._settings['skin'])
        self._cluster = Cluster(self._settings['cluster'])
        self._selector = Selector(self._settings['selector'])
        self._name = Name(samples, labels, self._settings['name'])

    def get(self, img):
        resized = self._resize.get(img)
        back_mask = self._back.get(resized)
        skin_mask = self._skin.get(resized)
        mask = back_mask | skin_mask
        k, labels, clusters_centers = self._cluster.get(resized[~mask])
        centers = self._selector.get(k, labels, clusters_centers)
        colors = [self._name.get(c) for c in centers]
        flattened = list({c for l in colors for c in l})

        if self._settings['debug'] is None:
            return flattened, centers[0] * 256  # centers * 256 added

        colored_labels = np.zeros((labels.shape[0], 3), np.float64)
        for i, c in enumerate(clusters_centers):
            colored_labels[labels == i] = c

        clusters = np.zeros(resized.shape, np.float64)
        clusters[~mask] = colored_labels

        return flattened, {
            'resized': resized,
            'back': back_mask,
            'skin': skin_mask,
            'clusters': clusters
        }

    @staticmethod
    def _default_settings():
        return {
            'resize': {},
            'back': {},
            'skin': {},
            'cluster': {},
            'selector': {},
            'name': {},
        }

    def get_dominant_color_by_dir(self, dir_image):
        img = skimage.io.imread(dir_image)
        color_name, color_rgb = self.get(img)
        return color_name, color_rgb

    def get_dominant_color_rgb(self, img):
        color_name, color_rgb = self.get(img)
        return color_rgb

    def get_dominant_color_text(self, img):
        color_name, color_rgb = self.get(img)
        return color_name

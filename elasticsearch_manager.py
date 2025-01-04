import os
import cv2
import numpy as np
import tensorflow as tf

from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from datetime import datetime
from elasticsearch import Elasticsearch

from vision.filters.color.dominant_color_algorithms.algolia.color_extractor.image_to_color import ImageToColor
from vision.object_detection.object_detection_yolo import YoloV3ObjectDetector
from vision.filters.texture.dominant_texture import lbp_histogram_rgb

CLASS_LABEL_FOLDER = 'models/object_detection/yolo-openimage-600/openimages.names'
models_dir = "./models"
dataset_dir_main = './dataset/'
model_list = ['InceptionResNetV2']


def get_image_crop(image_name, directory, image_detect, bounding_boxes_id, image_shape):
    left, top, width, height = image_detect['boxes'][bounding_boxes_id]
    right = left + width
    down = top + height
    if top < 0:
        top = 0
    if left < 0:
        left = 0

    image_complete = cv2.imread(directory + image_name)
    image_complete = cv2.resize(image_complete, (image_shape[1], image_shape[0]))
    image_crop = image_complete[top:down, left:right].copy()
    image_crop_name = 'crop_' + str(bounding_boxes_id) + '_' + image_name + '.jpg'
    cv2.imwrite(directory + image_crop_name, image_crop)
    return left, top, width, height, image_crop_name


class ElasticManager:
    def __init__(self, model_list):
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.330)
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, device_count={'GPU': 1})
        tf.compat.v1.Session(config=config)
        self.model_dict = {}

        if 'InceptionResNetV2' in model_list:
            weight_path = os.path.join(models_dir,
                                       'inception_resnet_v2'
                                       '/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')
            self.model_dict['InceptionResNetV2'] = self.model_InceptionResNetV2 = InceptionResNetV2(weights=weight_path,
                                                                                                    include_top=False,
                                                                                                    input_shape=(
                                                                                                        512, 512, 3),
                                                                                                    pooling='avg')
        self.object_detector = YoloV3ObjectDetector()
        self.es_client = Elasticsearch(
            ['192.168.112.89'],
            http_auth=('ma.eilbeigi', 'T0OQ1h0878'),
            scheme="https",
            port=9200,
            # ssl_context=None,
            verify_certs=False
        )
        self.index_number = self.find_number_index()
        self.image_to_color = ImageToColor()

    def find_number_index(self):
        es_count = self.es_client.count(index='cbir', doc_type='image', body={"query": {"match_all": {}}})['count']
        return es_count

    def get_model_embedding(self, model_name, img):
        img = image.load_img(img, target_size=(512, 512))
        x = image.img_to_array(img)
        x = preprocess_input(np.array([x]))
        return self.model_dict[model_name].predict(x)

    def generate_elastic_index(self, dataset_dir_main):
        iterate_id = 0
        actions = []
        image_err = []

        for directory in os.listdir(dataset_dir_main):
            directory = dataset_dir_main + directory + '/'
            print('directory', directory)
            for image_name in os.listdir(directory):
                print(iterate_id, '*', image_name)
                try:
                    if image_name.endswith(".jpg") or image_name.endswith(".png") or image_name.endswith(".jpeg"):
                        location_image = os.path.join(directory, image_name)

                        # detection
                        boxes, percentages, class_ids, image_main_shape = self.object_detector.detect(location_image)
                        class_ids = [open(CLASS_LABEL_FOLDER, "r").readlines()[i] for i in class_ids]
                        image_detect = {'boxes': boxes, 'class_ids': class_ids}

                        flag_crop = 1
                        number_crop_image = len(image_detect['boxes'])
                        if number_crop_image == 0:
                            number_crop_image = 1
                            flag_crop = 0
                        for bounding_boxes_id in range(number_crop_image):
                            if len(image_detect['boxes']) == 0:
                                left, top, width, height = image_main_shape[1], image_main_shape[0], 0, 0

                                # embedding
                                current_embedding = self.get_model_embedding('InceptionResNetV2', location_image)
                                # texture
                                dominant_texture_image = lbp_histogram_rgb(image_name, directory)
                                dominant_texture_image = dominant_texture_image.reshape((1, -1))
                                # color
                                # dominant_color = dominant_color_histogram(image_name, directory)
                                color_name, color_rgb = self.image_to_color.get_dominant_color_by_dir(location_image)
                                # elastic index
                                body = {
                                    # 'id': iterate_id,
                                    "image_name": image_name,
                                    "image_category": image_name[image_name.find('-') + 1: image_name.find('_')],
                                    "relative_path": directory,
                                    "image_width": image_main_shape[1],
                                    "image_height": image_main_shape[0],
                                    "object_width": width,
                                    "object_height": height,
                                    "bounding_box": [left, top, width, height],
                                    "irn2_embedding": current_embedding,
                                    "dominant_color_text": color_name,
                                    "dominant_color_rgb": color_rgb,
                                    "dominant_texture": dominant_texture_image,
                                    "detected_class": '-' if not flag_crop else class_ids[bounding_boxes_id].rstrip(),
                                    "deleted": False,
                                    "timestamp": datetime.now()
                                }
                                self.es_client.index(id=iterate_id, index='cbir', body=body, doc_type='image')
                                # actions.append(body)
                                iterate_id += 1
                            else:
                                left, top, width, height, image_name_crop = get_image_crop(image_name,
                                                                                           directory,
                                                                                           image_detect,
                                                                                           bounding_boxes_id,
                                                                                           image_main_shape)

                                location_image = os.path.join(directory, image_name_crop)
                                # embedding
                                current_embedding = self.get_model_embedding('InceptionResNetV2', location_image)
                                # texture
                                dominant_texture_image = lbp_histogram_rgb(image_name_crop, directory)
                                dominant_texture_image = dominant_texture_image.reshape((1, -1))
                                # color
                                # dominant_color = dominant_color_histogram(image_name_crop, directory)
                                color_name, color_rgb = self.image_to_color.get_dominant_color_by_dir(location_image)
                                # elastic index
                                body = {
                                    # 'id': iterate_id,
                                    "image_name": image_name,
                                    "image_category": image_name[image_name.find('-') + 1: image_name.find('_')],
                                    "relative_path": directory,
                                    "image_width": image_main_shape[1],
                                    "image_height": image_main_shape[0],
                                    "object_width": width,
                                    "object_height": height,
                                    "bounding_box": [left, top, width, height],
                                    "irn2_embedding": current_embedding,
                                    "dominant_color_text": color_name,
                                    "dominant_color_rgb": color_rgb,
                                    "dominant_texture": dominant_texture_image,
                                    "detected_class": '-' if not flag_crop else class_ids[bounding_boxes_id].rstrip(),
                                    "deleted": False,
                                    "timestamp": datetime.now()
                                }
                                self.es_client.index(id=iterate_id, index='cbir', body=body, doc_type='image')
                                # actions.append(body)
                                iterate_id += 1

                                os.remove(location_image)
                except:
                    print('err: ', location_image)
                    image_err.append(directory + ' ' + image_name)
                    with open('image_err.txt', 'w') as f:
                        f.write(str(image_err))
            # self.es_client.bulk(actions, index='cbir', doc_type='image')
        # print(actions)
        return self.es_client

    def extract_field_image(self, image_name, directory):
        actions = []
        location_image = os.path.join(directory, image_name)

        # detection
        boxes, percentages, class_ids, image_main_shape = self.object_detector.detect(location_image)
        class_ids = [open(CLASS_LABEL_FOLDER, "r").readlines()[i] for i in class_ids]
        image_detect = {'boxes': boxes, 'class_ids': class_ids}

        flag_crop = 1
        number_crop_image = len(image_detect['boxes'])
        if number_crop_image == 0:
            number_crop_image = 1
            flag_crop = 0
        for bounding_boxes_id in range(number_crop_image):
            if len(image_detect['boxes']) == 0:
                left, top, width, height = image_main_shape[1], image_main_shape[0], 0, 0

                # embedding
                current_embedding = self.get_model_embedding('InceptionResNetV2', location_image)
                # texture
                dominant_texture_image = lbp_histogram_rgb(image_name, directory)
                dominant_texture_image = dominant_texture_image.reshape((1, -1))
                # color
                # dominant_color = dominant_color_histogram(image_name, directory)
                color_name, color_rgb = self.image_to_color.get_dominant_color_by_dir(location_image)
                # elastic index
                body = {
                    # 'id': iterate_id,
                    "image_name": image_name,
                    "image_category": '-' if not flag_crop else class_ids[bounding_boxes_id].rstrip(),
                    "relative_path": directory,
                    "image_width": image_main_shape[1],
                    "image_height": image_main_shape[0],
                    "object_width": width,
                    "object_height": height,
                    "bounding_box": [left, top, width, height],
                    "irn2_embedding": current_embedding,
                    "dominant_color_text": color_name,
                    "dominant_color_rgb": color_rgb,
                    "dominant_texture": dominant_texture_image,
                    "detected_class": '-' if not flag_crop else class_ids[bounding_boxes_id].rstrip(),
                    "deleted": False,
                    "timestamp": datetime.now()
                }
                actions.append(body)
            else:
                left, top, width, height, image_name_crop = get_image_crop(image_name,
                                                                           directory,
                                                                           image_detect,
                                                                           bounding_boxes_id,
                                                                           image_main_shape)

                location_image = os.path.join(directory, image_name_crop)
                # embedding
                current_embedding = self.get_model_embedding('InceptionResNetV2', location_image)
                # texture
                dominant_texture_image = lbp_histogram_rgb(image_name_crop, directory)
                dominant_texture_image = dominant_texture_image.reshape((1, -1))
                # color
                # dominant_color = dominant_color_histogram(image_name_crop, directory)
                color_name, color_rgb = self.image_to_color.get_dominant_color_by_dir(location_image)
                # elastic index
                body = {
                    # 'id': iterate_id,
                    "image_name": image_name,
                    "image_category": '-' if not flag_crop else class_ids[bounding_boxes_id].rstrip(),
                    "relative_path": directory,
                    "image_width": image_main_shape[1],
                    "image_height": image_main_shape[0],
                    "object_width": width,
                    "object_height": height,
                    "bounding_box": [left, top, width, height],
                    "irn2_embedding": current_embedding,
                    "dominant_color_text": color_name,
                    "dominant_color_rgb": color_rgb,
                    "dominant_texture": dominant_texture_image,
                    "detected_class": '-' if not flag_crop else class_ids[bounding_boxes_id].rstrip(),
                    "deleted": False,
                    "timestamp": datetime.now()
                }
                actions.append(body)
                os.remove(location_image)
        return actions
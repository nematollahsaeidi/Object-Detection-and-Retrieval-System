import time

import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input
from keras.applications.xception import Xception, preprocess_input

models_dir = "./models"


class EmbeddingManager:
    def __init__(self, model_list):
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.130)
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, device_count={'GPU': 1})
        tf.compat.v1.Session(config=config)
        self.model_dict = {}

        if 'MobileNetV2' in model_list:
            weight_path = os.path.join(models_dir,
                                       'mobile_net_v2'
                                       '/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')
            self.model_dict['MobileNetV2'] = MobileNetV2(weights=weight_path,
                                                         include_top=False,
                                                         input_shape=(512, 512, 3),
                                                         pooling='avg')
        if 'InceptionResNetV2' in model_list:
            weight_path = os.path.join(models_dir,
                                       'inception_resnet_v2'
                                       '/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')
            self.model_dict['InceptionResNetV2'] = InceptionResNetV2(weights=weight_path,
                                                                     include_top=False,
                                                                     input_shape=(512, 512, 3),
                                                                     pooling='avg')

        if 'NASNetLarge' in model_list:
            weight_path = os.path.join(models_dir,
                                       'nas_net_large'
                                       '/NASNet-large-no-top.h5')
            self.model_dict['NASNetLarge'] = NASNetLarge(weights=weight_path,
                                                         include_top=False,
                                                         input_shape=(512, 512, 3),
                                                         pooling='avg')
        if 'Xception' in model_list:
            weight_path = os.path.join(models_dir,
                                       'xception'
                                       '/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
            self.model_dict['Xception'] = Xception(weights=weight_path,
                                                   include_top=False,
                                                   input_shape=(512, 512, 3),
                                                   pooling='avg')

    def get_model_embedding(self, model_name, img):
        x = image.img_to_array(img)
        x = preprocess_input(np.array([x]))
        return self.model_dict[model_name].predict(x)

    def generate_embeddings(self, dataset_dir_main):
        embeddings_db = {}
        for directory in os.listdir(dataset_dir_main):
            directory = dataset_dir_main + directory + '/'
            print('directory', directory)
            for file in os.listdir(directory):
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                    file_name = os.path.join(directory, file)
                    try:
                        img = image.load_img(file_name, target_size=(512, 512))
                    except OSError as e:
                        print(file, str(e))
                        continue
                    current_embedding = {}
                    for model_name in self.model_dict.keys():
                        s = time.time()
                        current_embedding[model_name] = self.get_model_embedding(model_name, img)
                        print(f"Embedding {model_name} for {file} takes {(time.time() - s):.3f}")
                    embeddings_db[file] = current_embedding
        return embeddings_db

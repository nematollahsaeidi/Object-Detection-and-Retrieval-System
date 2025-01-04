import time
import torch
from detectron2.utils.logger import setup_logger

setup_logger()

import cv2
import numpy as np
import shutil
import pickle
import os
import pandas as pd
import tensorflow as tf
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from sklearn.metrics import classification_report
from scipy.spatial import distance
from glob import glob
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.applications import InceptionResNetV2, NASNetLarge, MobileNetV2, Xception, NASNetMobile
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# from k_shot_learning.object_detection_yolo import YoloV3ObjectDetector

# OD = YoloV3ObjectDetector()
HEIGHT = 224
WIDTH = 224


def process_classes_boundingBoxes(model, folder_of_classes, features_folder_path):
    """
    Process available classes of images and save the feature map of every image
    and save average of all the feature maps of every class in .pkl format

    :param model:
    :param folder_of_classes:
    :return:
    """
    global HEIGHT, WIDTH

    # Iterate over class directories in originalData_FolderPath:
    originalData_ClassPaths = glob(folder_of_classes + '/*/')
    print('classes:', originalData_ClassPaths)
    print('Number of classes:', len(originalData_ClassPaths))

    for classPath in originalData_ClassPaths:

        className = classPath.split(os.path.sep)[-2]
        imagePaths = glob(classPath + '*.' + '*')  # List of all images in this special class

        get_value = False
        average_feature_map = 0

        print('Class', originalData_ClassPaths.index(classPath), ':', className, ') Number of images:', len(imagePaths))
        create_folder(features_folder_path + className)  # Create a folder to save the feature maps of this class

        # Extract the feature map of each image:
        for image_path in imagePaths:

            # print(image_path)
            image_name = image_path.split(os.path.sep)[-1].split('.')[-2]
            # print(image_name)

            start_time = time.time()
            # feature_map = detectron2_model_extract_feature_map(model=model, image_path=image_path, height=HEIGHT,
            #                                                    width=WIDTH)
            print('feature_map', time.time() - start_time)
            feature_map = detectron2_extract_feature_of_boundingBoxes(model=model, image_path=image_path, height=HEIGHT,
                                                                      width=WIDTH)
            # start_time = time.time()
            # feature_map = OD.detect(image_path)
            # print('feature_map', time.time() - start_time)
            if feature_map.shape[0] != 1:
                # Save the feature_map in pickle format in the features_folder_path:
                feature_map_fullPath = features_folder_path + className + '/' + image_name + '.pkl'
                with open(feature_map_fullPath, 'wb') as handle:  # pickling
                    pickle.dump(feature_map, handle)

        #     # Calculate Average:
        #     if not get_value:
        #         average_feature_map = feature_map
        #         get_value = True
        #     else:
        #         average_feature_map = average_feature_map + feature_map
        #
        # average_feature_map = average_feature_map / len(imagePaths)
        # # Save average_feature_map:
        # feature_map_fullPath = features_folder_path + className + '_' + 'average' + '.pkl'
        # with open(feature_map_fullPath, 'wb') as handle:  # pickling
        #     pickle.dump(average_feature_map, handle)
        # # print('=================================================================================')

    return


def cross_test(features_folder_path, method):
    """
    This function finds the most similar feature map to feature_map from the .pkl files
    in the class folders in features_folder_path and return the name of its class

    :param main_feature_map:
    :param features_folder_path:
    :return:
    """

    list_of_feature_maps_path = glob(features_folder_path + '/*/*.*')
    number_of_feature_maps = len(list_of_feature_maps_path)
    accuracy = 0
    max_true_distance = -10000  # max distance among all true predicts
    min_false_distance = 10000  # min distance among all wrong predicts
    y_true = []
    y_pred = []
    wrong_predicts_df = pd.DataFrame(columns=['Source_feature_map', 'MostSimilar_feature_map',
                                              'Source_class', 'MostSimilar_class'])
    #########ecom
    all_predicts_df = pd.DataFrame(columns=['Source_feature_map', 'MostSimilar_feature_map', 'Source_class',
                                            'MostSimilar_class', 'image_name_list_top_n', 'sorted_similarity_top_n'])
    #############

    print('Number of feature maps:', number_of_feature_maps)

    for feature_map_index in range(number_of_feature_maps):
        print('feature_map_index', feature_map_index)

        main_feature_map_path = list_of_feature_maps_path[feature_map_index]
        main_className = main_feature_map_path.split(os.path.sep)[-2]  # Class of the feature map
        with open(main_feature_map_path, "rb") as fp:  # Unpickling the main feature map:
            main_feature_map = pickle.load(fp)

        new_list_of_feature_maps_path = list_of_feature_maps_path.copy()  # With new_list = my_list, you don't actually have two lists. The assignment just copies the reference to the list, not the actual list
        del new_list_of_feature_maps_path[feature_map_index]

        # *******************************************************************************
        feature_map = 0
        minimum_feature_map = 0  # Feature map with minimum distance == similar feature map
        minimum_distance = 1000000000000
        get_value = False  # whether min_feature_map gets its first value or not
        most_similar_class = ''
        most_similar_feature_map_path = ''
        dist_lists = []
        for feature_map_path in new_list_of_feature_maps_path:

            className = feature_map_path.split(os.path.sep)[-2]  # Class of the feature map
            with open(feature_map_path, "rb") as fp:  # Unpickling the feature map:
                feature_map = pickle.load(fp)
            dist = compute_distance(main_feature_map, feature_map)
            dist_lists.append(dist)  # ecom
            if minimum_distance > dist:
                minimum_distance = dist
                minimum_feature_map = feature_map
                most_similar_feature_map_path = feature_map_path
                most_similar_class = className

        ###############ecom
        # from scipy.spatial.distance import cdist
        # cos_sim = 1 - cdist(image_embeddings, self.db_embeddings, metric='cosine')
        # cos_sim = cos_sim.ravel()
        sorted_similarity = np.argsort(dist_lists, axis=0)
        image_name_list = [new_list_of_feature_maps_path[i] for i in sorted_similarity]
        image_name_list_top_n = image_name_list[0: 5]
        sorted_similarity_top_n = np.positive(np.sort(dist_lists, axis=0))[0:5]
        sorted_simi_top_n = sorted_similarity[0:5]
        all_predicts_df = all_predicts_df.append({'Source_feature_map': main_feature_map_path,
                                                  'MostSimilar_feature_map': most_similar_feature_map_path,
                                                  'Source_class': main_className,
                                                  'MostSimilar_class': most_similar_class,
                                                  'image_name_list_top_n': image_name_list_top_n,
                                                  'sorted_similarity_top_n': sorted_similarity_top_n},
                                                 ignore_index=True)
        ###############
        y_true.append(main_className)
        y_pred.append(most_similar_class)

        if main_className == most_similar_class:
            accuracy += 1
            if max_true_distance < minimum_distance:
                max_true_distance = minimum_distance
        else:
            wrong_predicts_df = wrong_predicts_df.append({'Source_feature_map': main_feature_map_path,
                                                          'MostSimilar_feature_map': most_similar_feature_map_path,
                                                          'Source_class': main_className,
                                                          'MostSimilar_class': most_similar_class},
                                                         ignore_index=True)
            if min_false_distance > minimum_distance:
                min_false_distance = minimum_distance

    wrong_predicts_df.to_csv('results/wrong_predicts_df.csv', index=False)
    all_predicts_df.to_csv('results/' + method + '_all_predicts_df.csv', index=False)  ##########ecom
    accuracy /= number_of_feature_maps

    classification_report_dict = classification_report(y_true, y_pred, output_dict=True)
    classification_report_df = pd.DataFrame(classification_report_dict)

    print('accuracy=', accuracy)
    print('max_true_distance=', max_true_distance)
    print('min_false_distance=', min_false_distance)

    print(classification_report(y_true, y_pred))
    classification_report_df.to_csv('results/classification_report_df.csv', index=True)
    return accuracy


def create_folder(path):
    # Remove the folder if exists:
    path = Path(path)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    # Crete the folder:
    if not os.path.exists(path):
        os.makedirs(path)

    return


def compute_distance(vector1, vector2):
    # if len(vector1)==1 or len(vector2)==1 :
    #     return -1

    dist = distance.cosine(vector1, vector2)

    return dist


def detectron2_extract_feature_of_boundingBoxes(model, image_path, height, width):
    """
    # https://detectron2.readthedocs.io/tutorials/models.html#partially-execute-a-model
    # https://stackoverflow.com/questions/62442039/detectron2-extract-region-features-at-a-threshold-for-object-detection
    """

    image = cv2.imread(image_path)

    height, width = image.shape[:2]
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width}]

    with torch.no_grad():
        images = model.preprocess_image(inputs)  # don't forget to preprocess
        features = model.backbone(images.tensor)  # set of cnn features
        proposals, _ = model.proposal_generator(images, features, None)  # RPN

        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
        predictions = model.roi_heads.box_predictor(box_features)
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

        # output boxes, masks, scores, etc
        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
        # features of the proposed boxes
        feats = box_features[pred_inds]

    # # print('************************************************')
    # # for k in pred_instances[0]["instances"].pred_classes:
    # #   print(k)
    # # for k in pred_instances[0]["instances"].pred_boxes:
    # #   print(k)
    # # print('************************************************')
    # print('Number of boxes:', len(feats))  # Number of boxes
    # # print(feats[0].shape)  # shape of feature map for box 0
    # print('# ***********************************************************')
    # # Feature Map:
    # # print(feats[0])  # Feature of box 0
    # print('# ***********************************************************')

    if len(feats) == 0:
        feature_map = np.array([-1])
    else:
        feature_map = feats[0]  # the first bounding box
        feature_map = feature_map.cpu().numpy()
    return feature_map


def keras_extract_feature_map(model, image_path, height=HEIGHT, width=WIDTH):
    """
    Extract the feature map from an image (whole image -> 1 feature map)

    :param model:
    :param image_path:
    :param height:
    :param width:
    :return:
    """

    image = cv2.imread(image_path)
    # print(type(image))
    # print(image.shape)
    image = cv2.resize(image, dsize=(height, width))  # , interpolation=cv2.INTER_CUBIC)
    # print(image.shape)

    height, width = image.shape[:2]
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width}]

    with torch.no_grad():
        images = model.preprocess_image(inputs)  # don't forget to preprocess
        features = model.backbone(images.tensor)  # set of cnn features
        proposals, _ = model.proposal_generator(images, features, None)  # RPN

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # for key in features.keys():
        #     print(key)
        # feature_map = features['p2']  #################################
        feature_map = features['res4']
        # print(feature_map.shape)

        feature_map = feature_map.cpu().numpy()
        feature_map = feature_map.reshape(-1)

    #
    #     features_ = [features[f] for f in model.roi_heads.box_in_features]
    #     box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
    #     box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
    #     predictions = model.roi_heads.box_predictor(box_features)
    #     pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
    #     pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)
    #
    #     # output boxes, masks, scores, etc
    #     pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
    #     # features of the proposed boxes
    #     feats = box_features[pred_inds]
    #
    # # print('************************************************')
    # # for k in pred_instances[0]["instances"].pred_classes:
    # #   print(k)
    # # for k in pred_instances[0]["instances"].pred_boxes:
    # #   print(k)
    # # print('************************************************')
    # print('Number of boxes:', len(feats))  # Number of boxes
    # # print(feats[0].shape)  # shape of feature map for box 0
    # print('# ***********************************************************')
    # # Feature Map:
    # # print(feats[0])  # Feature of box 0
    # print('# ***********************************************************')

    return feature_map


def detectron2_model_extract_feature_map(model, image_path, height=HEIGHT, width=WIDTH):
    """
    Extract the feature map from an image (whole image -> 1 feature map)

    :param model:
    :param image_path:
    :param height:
    :param width:
    :return:
    """

    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=(height, width))  # , interpolation=cv2.INTER_CUBIC)

    height, width = image.shape[:2]
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width}]

    with torch.no_grad():
        images = model.preprocess_image(inputs)  # don't forget to preprocess
        features = model.backbone(images.tensor)  # set of cnn features
        proposals, _ = model.proposal_generator(images, features, None)  # RPN

        # for key in features.keys():
        #     print(key)
        # feature_map = features['p2']  #################################
        feature_map = features['res4']
        # print(feature_map.shape)

        feature_map = feature_map.cpu().numpy()
        feature_map = feature_map.reshape(-1)
        # print(feature_map.shape)

    #
    #     features_ = [features[f] for f in model.roi_heads.box_in_features]
    #     box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
    #     box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
    #     predictions = model.roi_heads.box_predictor(box_features)
    #     pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
    #     pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)
    #
    #     # output boxes, masks, scores, etc
    #     pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
    #     # features of the proposed boxes
    #     feats = box_features[pred_inds]
    #
    # # print('************************************************')
    # # for k in pred_instances[0]["instances"].pred_classes:
    # #   print(k)
    # # for k in pred_instances[0]["instances"].pred_boxes:
    # #   print(k)
    # # print('************************************************')
    # print('Number of boxes:', len(feats))  # Number of boxes
    # # print(feats[0].shape)  # shape of feature map for box 0
    # print('# ***********************************************************')
    # # Feature Map:
    # # print(feats[0])  # Feature of box 0
    # print('# ***********************************************************')
    return feature_map


def process_classes(model, folder_of_classes, features_folder_path):
    """
    Process available classes of images and save the feature map of every image
    and save average of all the feature maps of every class in .pkl format

    :param model:
    :param folder_of_classes:
    :return:
    """
    global HEIGHT, WIDTH

    # Iterate over class directories in originalData_FolderPath:
    originalData_ClassPaths = glob(folder_of_classes + '/*/')
    print('classes:', originalData_ClassPaths)
    print('Number of classes:', len(originalData_ClassPaths))

    for classPath in originalData_ClassPaths:

        className = classPath.split(os.path.sep)[-2]
        imagePaths = glob(classPath + '*.' + '*')  # List of all images in this special class

        get_value = False
        average_feature_map = 0

        print('Class', originalData_ClassPaths.index(classPath), ':', className, ') Number of images:', len(imagePaths))
        create_folder(features_folder_path + className)  # Create a folder to save the feature maps of this class

        # Extract the feature map of each image:
        for image_path in imagePaths:

            # print(image_path)
            image_name = image_path.split(os.path.sep)[-1].split('.')[-2]
            # print(image_name)

            feature_map = detectron2_model_extract_feature_map(model=model, image_path=image_path, height=HEIGHT,
                                                               width=WIDTH)

            # Save the feature_map in pickle format in the features_folder_path:
            feature_map_fullPath = features_folder_path + className + '/' + image_name + '.pkl'
            with open(feature_map_fullPath, 'wb') as handle:  # pickling
                pickle.dump(feature_map, handle)

            # Calculate Average:
            if not get_value:
                average_feature_map = feature_map
                get_value = True
            else:
                average_feature_map = average_feature_map + feature_map

        average_feature_map = average_feature_map / len(imagePaths)
        # Save average_feature_map:
        feature_map_fullPath = features_folder_path + className + '_' + 'average' + '.pkl'
        with open(feature_map_fullPath, 'wb') as handle:  # pickling
            pickle.dump(average_feature_map, handle)
        # print('=================================================================================')
    return


def find_similar_feature_map(main_feature_map, features_folder_path):
    """
    This function find the most similar feature map to feature_map from the .pkl files
    in the class folders in features_folder_path and return the name of its class

    :param main_feature_map:
    :param features_folder_path:
    :return:
    """
    print('# ***********************************************************')
    print('find_similar_feature_map:')

    feature_map = 0
    minimum_feature_map = 0  # Feature map with minimum distance == similar feature map
    minimum_distance = 100000
    get_value = False  # whether min_feature_map gets its first value or not
    most_similar_class = ''

    list_of_feature_maps_path = glob(features_folder_path + '/*/*.*')

    for feature_map_path in list_of_feature_maps_path:

        className = feature_map_path.split(os.path.sep)[-2]  # Class of the feature map
        with open(feature_map_path, "rb") as fp:  # Unpickling the feature map:
            feature_map = pickle.load(fp)

        dist = compute_distance(main_feature_map, feature_map)

        if minimum_distance > dist:
            minimum_distance = dist
            minimum_feature_map = feature_map
            most_similar_class = className

    return most_similar_class, minimum_distance


def find_similar_feature_map_from_averages(main_feature_map, features_folder_path):
    """
    This function find the most similar feature map to feature_map from the .pkl files
    in the class folders in features_folder_path and return the name of its class

    :param main_feature_map:
    :param features_folder_path:
    :return:
    """
    print('# ***********************************************************')
    print('find_similar_feature_map_from_averages:')

    feature_map = 0
    minimum_feature_map = 0  # Feature map with minimum distance == similar feature map
    minimum_distance = 100000
    get_value = False  # whether min_feature_map gets its first value or not
    most_similar_class = ''

    list_of_feature_maps_path = glob(features_folder_path + '*.*')
    # print(list_of_feature_maps_path)

    for feature_map_path in list_of_feature_maps_path:

        className = feature_map_path.split(os.path.sep)[-1].split('_')[-2]  # Class of the feature map
        with open(feature_map_path, "rb") as fp:  # Unpickling the feature map:
            feature_map = pickle.load(fp)

        dist = compute_distance(main_feature_map, feature_map)

        if minimum_distance > dist:
            minimum_distance = dist
            minimum_feature_map = feature_map
            most_similar_class = className

    return most_similar_class, minimum_distance


def detect_class_of_image(model, image_path, features_folder_path, height=HEIGHT, width=WIDTH):
    """
    :param model:
    :param image_path:
    :param features_folder_path:
    :param height:
    :param width:
    :return:
    """

    global HEIGHT, WIDTH

    feature_map = detectron2_model_extract_feature_map(model, image_path, height=HEIGHT, width=WIDTH)

    most_similar_class, minimum_distance = find_similar_feature_map(main_feature_map=feature_map,
                                                                    features_folder_path=features_folder_path)

    most_similar_average_class, minimum_average_distance = find_similar_feature_map_from_averages(
        main_feature_map=feature_map,
        features_folder_path=features_folder_path)

    print('most_similar_class:', most_similar_class, ') minimum_distance:', minimum_distance)
    print('most_similar_average_class:', most_similar_average_class, ') minimum_average_distance:',
          minimum_average_distance)
    return


def get_tensorflow_imagenet_pretrained_model(inputShape=(HEIGHT, WIDTH, 3)):
    # Create base models:
    # Create the base model from the pre-trained model DenseNet:
    # base_model_1 = DenseNet201(
    #     include_top=False,
    #     weights='imagenet',
    #     input_tensor=None,
    #     input_shape=inputShape,
    #     pooling=None,
    #     # classes=numberOfClasses
    # )
    #
    # base_model_2 = VGG19(
    #     include_top=False,
    #     weights="imagenet",
    #     input_tensor=None,
    #     input_shape=inputShape,
    #     pooling=None,
    # )
    #
    base_model_3 = NASNetMobile(  # EfficientNetB7(  # NASNetMobile(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=inputShape,
        pooling=None,
    )

    base_model_4 = InceptionResNetV2(
        include_top=False,
        weights="/home/saeedi/PycharmProjects/visually-similar-search/k_shot_learning/model_keras/inception_resnet_v2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5",
        # weights='imagenet',
        input_tensor=None,
        input_shape=inputShape,
        pooling=None,
    )

    # base_model_5 = InceptionV3(
    #     include_top=False,
    #     weights="imagenet",
    #     input_tensor=None,
    #     input_shape=inputShape,
    #     pooling=None,
    # )
    #
    base_model_6 = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=inputShape,
        pooling=None,
    )
    #
    # base_model_7 = ResNet152V2(
    #     include_top=False,
    #     weights="imagenet",
    #     input_tensor=None,
    #     input_shape=inputShape,
    #     pooling=None,
    # )
    #
    base_model_8 = Xception(
        include_top=False,
        weights="/home/saeedi/PycharmProjects/visually-similar-search/k_shot_learning/model_keras/xception"
                "/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
        input_tensor=None,
        input_shape=inputShape,
        pooling=None,
    )

    base_model_9 = NASNetLarge(
        include_top=False,
        weights="/home/saeedi/PycharmProjects/visually-similar-search/k_shot_learning/model_keras/nas_net_large/"
                "NASNet-large-no-top.h5",
        input_tensor=None,
        input_shape=inputShape,
        pooling=None)
    # include_top = False,
    # weights = "imagenet",
    # input_tensor = None,
    # input_shape = inputShape,
    # pooling = None,
    # classes=100

    # list_of_base_models = [base_model_1, base_model_2, base_model_3, base_model_4,
    #                        base_model_5, base_model_6, base_model_7, base_model_8]
    list_of_base_models = [base_model_9]  #####################################################3

    for base_model in list_of_base_models:
        base_model.trainable = False

    inputs = keras.Input(shape=inputShape)

    # y = base_model_1(inputs)
    # y1 = Flatten()(y)
    #
    # y = base_model_2(inputs)
    # y2 = Flatten()(y)
    #
    y = base_model_3(inputs)
    y3 = Flatten()(y)

    y = base_model_4(inputs)
    y4 = Flatten()(y)

    # y = base_model_5(inputs)
    # y5 = Flatten()(y)
    #
    y = base_model_6(inputs)
    y6 = Flatten()(y)
    #
    # y = base_model_7(inputs)
    # y7 = Flatten()(y)
    #
    y = base_model_8(inputs)
    y8 = Flatten()(y)

    y = base_model_9(inputs)
    y9 = Flatten()(y)

    model = keras.Model(inputs=inputs, outputs=y9)  ############################################

    return model


def tensorflow_pretrained_models_extract_feature_map(model, image_path, height=HEIGHT, width=WIDTH):
    """
    Extract the feature map from an image (whole image -> 1 feature map)

    :param model:
    :param image_path:
    :param height:
    :param width:
    :return:
    """

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(height, width))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    feature_map = model.predict(x)
    # print(feature_map.shape)  # (1, 94080)
    feature_map = feature_map[0]  # (94080,)
    # print(feature_map.shape)

    feature_map = feature_map.reshape(-1)

    return feature_map


def tensorflow_imagenet_process_classes(model, folder_of_classes, features_folder_path):
    """
    Process available classes of images and save the feature map of every image
    and save average of all the feature maps of every class in .pkl format

    :param model:
    :param folder_of_classes:
    :return:
    """
    global HEIGHT, WIDTH

    # Iterate over class directories in originalData_FolderPath:
    originalData_ClassPaths = glob(folder_of_classes + '/*/')
    print(originalData_ClassPaths)

    for classPath in originalData_ClassPaths:

        className = classPath.split(os.path.sep)[-2]
        imagePaths = glob(classPath + '*.' + '*')  # List of all images in this special class

        get_value = False
        average_feature_map = 0

        print('Class: ', className, ') Number of images:', len(imagePaths))
        create_folder(features_folder_path + className)  # Create a folder to save the feature maps of this class

        # Extract the feature map of each image:
        for image_path in imagePaths:

            # print(image_path)
            image_name = image_path.split(os.path.sep)[-1].split('.')[-2]
            # print(image_name)
            start_time = time.time()
            feature_map = tensorflow_pretrained_models_extract_feature_map(model=model, image_path=image_path,
                                                                           height=HEIGHT,
                                                                           width=WIDTH)
            finish_time = time.time() - start_time
            print(finish_time)
            # Save the feature_map in pickle format in the features_folder_path:
            feature_map_fullPath = features_folder_path + className + '/' + image_name + '.pkl'
            with open(feature_map_fullPath, 'wb') as handle:  # pickling
                pickle.dump(feature_map, handle)

            # Calculate Average:
            if not get_value:
                average_feature_map = feature_map
                get_value = True
            else:
                average_feature_map = average_feature_map + feature_map

        average_feature_map = average_feature_map / len(imagePaths)
        # Save average_feature_map:
        feature_map_fullPath = features_folder_path + className + '_' + 'average' + '.pkl'
        with open(feature_map_fullPath, 'wb') as handle:  # pickling
            pickle.dump(average_feature_map, handle)
        print('=================================================================================')

    return


def tensorflow_imagenet_detect_class_of_image(model, image_path, features_folder_path, height=HEIGHT, width=WIDTH):
    """
    :param model:
    :param image_path:
    :param features_folder_path:
    :param height:
    :param width:
    :return:
    """

    global HEIGHT, WIDTH

    feature_map = tensorflow_pretrained_models_extract_feature_map(model, image_path, height=HEIGHT, width=WIDTH)

    most_similar_class, minimum_distance = find_similar_feature_map(main_feature_map=feature_map,
                                                                    features_folder_path=features_folder_path)

    most_similar_average_class, minimum_average_distance = find_similar_feature_map_from_averages(
        main_feature_map=feature_map,
        features_folder_path=features_folder_path)

    print('most_similar_class:', most_similar_class, ') minimum_distance:', minimum_distance)
    print('most_similar_average_class:', most_similar_average_class, ') minimum_average_distance:',
          minimum_average_distance)
    return


def main():
    global HEIGHT, WIDTH
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml"))
    # cfg.MODEL.cfg = "/home/saeedi/PycharmProjects/visually-similar-search/few-shot-object-detection-master/configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml"
    # cfg.MODEL.cfg = "/home/saeedi/PycharmProjects/visually-similar-search/few-shot-object-detection-master/configs/LVIS-detection/faster_rcnn_R_101_FPN_fc_novel.yaml"
    # cfg.MODEL.cfg ="/home/saeedi/PycharmProjects/visually-similar-search/few-shot-object-detection-master/configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml"
    cfg.MODEL.cfg = "/home/saeedi/PycharmProjects/visually-similar-search/few-shot-object-detection-master/configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml")
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_10shot_unfreeze.yaml")
    # cfg.MODEL.WEIGHTS = "/home/saeedi/.torch/fvcore_cache/fs-det/models/voc/split1/tfa_cos_10shot/model_final.pth"
    # cfg.MODEL.WEIGHTS = "/home/saeedi/PycharmProjects/visually-similar-search/k_shot_learning/fs-det/models/lvis/R_101_FPN_repeat_fc/model_final.pth"
    # cfg.MODEL.WEIGHTS ="/home/saeedi/PycharmProjects/visually-similar-search/k_shot_learning/fs-det/models/coco/tfa_cos_30shot/model_final.pth"
    "/home/saeedi/PycharmProjects/visually-similar-search/k_shot_learning/fs-det/models/coco/tfa_cos_1shot/model_final.pth"
    # "/home/saeedi/PycharmProjects/visually-similar-search/k_shot_learning/fs-det/models/coco/FRCN+ft-full_30shot/model_final.pth"
    predictor = DefaultPredictor(cfg)
    # outputs = predictor(im)

    # build model
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()  # make sure its in eval mode

    print(cfg.MODEL.cfg)
    print(cfg.MODEL.WEIGHTS)

    all_images_path = 'images/'
    folder_of_classes_name = 'images_for_train/'  # 'images_v1/'  'images_for_train/'
    images_folder_path = all_images_path + folder_of_classes_name
    features_folder_path = 'feature_maps/' + folder_of_classes_name  # need to be created
    image_suffix = 'jpg'

    create_folder(features_folder_path)  # All the features should be saved here
    # process_classes(model=model, folder_of_classes=images_folder_path, features_folder_path=features_folder_path)
    process_classes_boundingBoxes(model=model, folder_of_classes=images_folder_path,
                                  features_folder_path=features_folder_path)
    # # Candidate image:
    # image_path = all_images_path + 'bag1.jpeg'
    # detect_class_of_image(model, image_path, features_folder_path, height=HEIGHT, width=WIDTH)

    # # ********************************************************************
    # print('Tensorflow pretrained models on imagenet:\n')

    # tensorflow pre-trained models on imagenet:
    # tensorflow_pretrained_model = get_tensorflow_imagenet_pretrained_model()
    # tensorflow_imagenet_process_classes(model=tensorflow_pretrained_model,
    #                                     folder_of_classes=images_folder_path,
    #                                     features_folder_path=features_folder_path)

    # Candidate image:
    # list_of_test_images = ['test1.jpg', 'test2.jpg', 'test3.jpg']
    # image_path = all_images_path + list_of_test_images[1]
    # tensorflow_imagenet_detect_class_of_image(tensorflow_pretrained_model, image_path, features_folder_path, height=HEIGHT, width=WIDTH)
    # # ********************************************************************

    print('Cross test:')
    start_time = time.time()
    method = 'COCO_faster_rcnn_R_101_FPN_fc_1shot'  ############################ecom
    cross_test(features_folder_path, method)
    finish_time = time.time() - start_time
    print(method)
    print(finish_time)
    return


# if __name__ == "__main__":
#     main()


class Detectron2ObjectDetector:
    def __init__(self):
        global HEIGHT, WIDTH
        cfg = get_cfg()
        # 'LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml'
        # "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml"))
        # cfg.MODEL.cfg = "/home/saeedi/PycharmProjects/visually-similar-search/few-shot-object-detection-master/configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml"
        # cfg.MODEL.cfg = "/home/saeedi/PycharmProjects/visually-similar-search/few-shot-object-detection-master/configs/LVIS-detection/faster_rcnn_R_101_FPN_fc_novel.yaml"
        # cfg.MODEL.cfg ="/home/saeedi/PycharmProjects/visually-similar-search/few-shot-object-detection-master/configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml"
        # cfg.MODEL.cfg = "/home/saeedi/PycharmProjects/visually-similar-search/few-shot-object-detection-master/configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_10shot_unfreeze.yaml")
        # cfg.MODEL.WEIGHTS = "/home/saeedi/.torch/fvcore_cache/fs-det/models/voc/split1/tfa_cos_10shot/model_final.pth"
        # cfg.MODEL.WEIGHTS = "/home/saeedi/PycharmProjects/visually-similar-search/k_shot_learning/fs-det/models/lvis/R_101_FPN_repeat_fc/model_final.pth"
        # cfg.MODEL.WEIGHTS ="/home/saeedi/PycharmProjects/visually-similar-search/k_shot_learning/fs-det/models/coco/tfa_cos_30shot/model_final.pth"
        # "/home/saeedi/PycharmProjects/visually-similar-search/k_shot_learning/fs-det/models/coco/tfa_cos_1shot/model_final.pth"
        # "/home/saeedi/PycharmProjects/visually-similar-search/k_shot_learning/fs-det/models/coco/FRCN+ft-full_30shot/model_final.pth"
        self.predictor = DefaultPredictor(cfg)

    def detect(self, image_path):
        im = cv2.imread(image_path)
        # img = cv2.resize(im, (500, 500))
        # height, width, channels = im.shape
        # from io_modules.disk.image_reader import image_read_resize
        # im = image_read_resize(image_path)  # for resize 0.8 image for images larger than 1200
        outputs = self.predictor(im)
        output_classes = outputs["instances"].pred_classes
        output_classes = output_classes.cpu().detach().numpy()
        output_boxes = outputs["instances"].pred_boxes
        output_boxes = output_boxes.tensor.cpu().detach().numpy()
        print(output_classes, output_boxes)

        image_shape = im.shape

        return output_classes, output_boxes, image_shape

    def get_names(self, class_ids):
        label_names = 'models/object_detection/coco/coco.names'
        return [open(label_names, "r").readlines()[i] for i in class_ids]

# image_dir = '/home/saeedi/PycharmProjects/visual_search/object_detection_service/im (15).jpg'
# D2OD = Detectron2ObjectDetector()
# output_pred_classes, output_pred_boxes = D2OD.detect(image_dir)
# print("Done")

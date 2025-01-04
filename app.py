import configparser
import cv2
import datetime
import json
import os
import glob
import numpy as np
import requests
import time
import skimage.io
from flask import Flask, request, redirect, url_for, render_template, jsonify, session
from werkzeug.utils import secure_filename
from urllib.parse import urlparse
from elasticsearch_manager import ElasticManager
from io_modules.disk.file_writer import write_file
from io_modules.disk.image_reader import get_gray_image
from io_modules.disk.request_utils import initiate_request, RequestDirectoryManager
from utils.log.logger import Logger
from keras.preprocessing import image
from utils.log.message_builder import build_message
from vision.dataset_loader.dataset_loader import DatasetLoader
from vision.filters.color.reranker.algolia_reranker import AlgoliaReranker
from vision.filters.texture.dominant_texture import rank_images_texture
from vision.indexing.embedding_manager import EmbeddingManager
from vision.object_detection.object_detection_yolo import YoloV3ObjectDetector
from vision.visual_recommender import VisualRecommender

config = configparser.ConfigParser()
config.read('cfg/config.cfg')

proxy = config.get('settings', 'proxy')
request_dir = config.get('dirs', 'request_dir')
dataset_folder = config.get('dirs', 'dataset_folder')
upload_folder = config.get('dirs', 'upload_folder')
static_folder = config.get('dirs', 'static_folder')
logs_path = config.get('dirs', 'logs_path')
location_image_web = config.get('dirs', 'location_image_web')
recommends_per_model = int(config.get('settings', 'number_recommend'))
# model_list = ['InceptionResNetV2']
model_list = ['NASNetLarge', 'ا']

directories_images = './dataset'
other_images = './dataset/other'

logger = Logger(logs_path).get_logger(__name__)

if not os.path.exists(request_dir):
    os.mkdir(request_dir)

app = Flask(__name__)
app.secret_key = config.get('settings', 'secret_key')
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = int(config.get('settings', 'MAX_CONTENT_LENGTH'))

embedding_manager = EmbeddingManager(model_list)
loader = DatasetLoader(embedding_manager, model_list)
visual_recommender = VisualRecommender(model_list, loader.dataset, loader.faiss_dict, embedding_manager)
object_detector = YoloV3ObjectDetector()
request_dir_manager = RequestDirectoryManager(request_dir)
es = ElasticManager(model_list)
color_reranker = AlgoliaReranker()


@app.route('/receive_image', methods=['GET'])
def receive_image():
    identity, request_time_string = initiate_request(request)
    req_data = request.get_json(force=True)
    url = req_data['image']
    try:
        message = build_message(identity, None, f"Request received: {str(req_data)}")
        logger.debug(json.dumps(message))
        start_time = time.time()
        response = requests.get(url, proxies={'http': proxy, 'https': proxy})
        message = build_message(identity, None, f"Download was done in {(time.time() - start_time):.3f} seconds.")
        logger.info(json.dumps(message))
        image_name = url.split('/')[-1]

        single_request_dir = request_dir_manager.create_dir_for_request(request_time_string)
        image_path_name = os.path.join(single_request_dir, image_name)
        write_file(image_path_name, response.content)

        image_path_name = os.path.dirname(image_path_name) + '/'
        actions_image = es.extract_field_image(image_name, image_path_name)

        index_image_bounding_box(actions_image, image_name, response)

        return jsonify({'results': message}), 200

    except (requests.ConnectionError, requests.exceptions.ReadTimeout) as e:
        status_code = 502
        msg = "Connection Error: I couldn't download URL."
        message = build_message(identity, status_code, msg, str(e))
        logger.exception(json.dumps(message))
        return jsonify({'results': message}), 502
    except Exception as e:
        status_code = 500
        msg = "Something went wrong."
        message = build_message(identity, status_code, msg, str(e))
        logger.exception(json.dumps(message))
        return jsonify({'results': message}), 500


def index_image_bounding_box(actions_image, image_name, response):
    index_number = es.find_number_index()
    iterate_id = index_number
    for action_image in actions_image:
        category_specific = action_image['image_category']
        if category_specific in os.listdir(directories_images):
            image_path_new = os.path.join(directories_images, category_specific, image_name)
            write_file(image_path_new, response.content)
        elif action_image['image_category'] == '-':
            image_path_new = os.path.join(other_images, image_name)
            write_file(image_path_new, response.content)
        else:
            new_category = os.path.join(directories_images, action_image['image_category'])
            os.mkdir(new_category)
            image_path_new = os.path.join(directories_images, category_specific, image_name)
            write_file(image_path_new, response.content)

        es.es_client.index(id=iterate_id, index='cbir', body=action_image, doc_type='image')
        iterate_id += 1
    return iterate_id


@app.route('/visual_search', methods=['GET'])
def visual_search():
    identity, request_time_string = initiate_request(request)
    req_data = request.get_json(force=True)
    url = req_data['image']
    try:
        message = build_message(identity, None, f"request is: {str(req_data)}")
        logger.debug(json.dumps(message))
        start_time = time.time()
        response = requests.get(url, proxies={'http': proxy, 'https': proxy})
        message = build_message(identity, None, f"Download was done in {(time.time() - start_time):.3f} seconds")
        logger.info(json.dumps(message))
        image_name = url.split('/')[-1]

        single_request_dir = request_dir_manager.create_dir_for_request(request_time_string)
        image_path_name = os.path.join(single_request_dir, image_name)
        a = datetime.datetime.now()
        write_file(image_path_name, response.content)
        b = datetime.datetime.now()
        message = build_message(identity, None, f"Store image in : {str(b - a)}")
        logger.info(json.dumps(message))

        message = build_message(identity, None, f"image_name: {image_name}")
        logger.debug(json.dumps(message))

        loaded_image = image.load_img(image_path_name, target_size=(512, 512))

        most_similar = visual_recommender.get_top_n(
            models=model_list,
            img=loaded_image,
            number_recommend=recommends_per_model
        )
        message = build_message(identity, None, f"most_similar: {most_similar}")
        logger.debug(json.dumps(message))
        if not most_similar:
            status_code = 404
            msg = f"With probability, your object is ," \
                  f"so your image request does not exist in this dataset."
            message = build_message(identity, status_code, msg, "your request don't exist in this dataset.")
            logger.error(json.dumps(message))
            return jsonify({'results': msg}), 404
        else:
            status_code = 200
            message = build_message(identity, status_code, f"Your request exists in this dataset.")
            logger.info(json.dumps(message))
            return jsonify({'results': most_similar}), 200
    except (requests.ConnectionError, requests.exceptions.ReadTimeout) as e:
        status_code = 502
        msg = "Connection Error: I couldn't download URL."
        message = build_message(identity, status_code, msg, str(e))
        logger.exception(json.dumps(message))
        return jsonify({'results': message}), 502
    except Exception as e:
        status_code = 500
        msg = "Something went wrong."
        message = build_message(identity, status_code, msg, str(e))
        logger.exception(json.dumps(message))
        return jsonify({'results': message}), 500


@app.route('/')
def upload_form():
    return render_template('master_page.html')


@app.route('/', methods=['POST'])
def upload_image():
    identity, request_time_string = initiate_request(request)
    check_color_bounding_box = 'False'
    selected_value_dominant = 'False'
    selected_distance = 'False'
    try:
        file = request.files['file']
        filename = secure_filename(file.filename)
        session['filename'] = filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_dir = upload_folder + filename
        boxes, percentages, class_ids, image_shape = object_detector.detect(image_dir)

        session['image_shape'] = image_shape

        areas = [-(i[2] + i[0]) * (i[3] + i[1]) for i in boxes]
        sorted_areas = np.argsort(areas, axis=0)
        boxes = [boxes[i] for i in sorted_areas]
        class_ids = [class_ids[i] for i in sorted_areas]

        class_names = object_detector.get_names(class_ids)
        detected_objects = {'boxes': boxes, 'class_ids': class_names}
        session['image_detect'] = detected_objects
        message = build_message(identity, None, f"detected_objects:{detected_objects}")
        logger.debug(json.dumps(message))
        image_name = filename
        if request.form.getlist('dominant') is not None:
            selected_value_dominant = request.form.getlist('dominant')
            session['selected_value_dominant'] = selected_value_dominant
        selected_distance = request.form['distance']
        session['selected_value_distance'] = selected_distance

    except Exception as e:
        message = build_message(identity, None, f"An exception has been occurred!", str(e))
        logger.debug(json.dumps(message))
        bounding_boxes_id = int(request.form['id'])
        filename = session.get('filename', None)
        detected_objects = session.get('image_detect', None)
        left, top, width, height = detected_objects['boxes'][bounding_boxes_id]
        right = left + width
        down = top + height
        if top < 0:
            top = 0
        if left < 0:
            left = 0
        message = build_message(identity, None, f"left: {left}, top:{top}, right: {right}, down:{down}")
        logger.debug(json.dumps(message))

        image_complete = cv2.imread(upload_folder + filename)
        image_shape = session.get('image_shape', None)
        image_complete = cv2.resize(image_complete, (image_shape[1], image_shape[0]))
        image_crop = image_complete[top:down, left:right].copy()
        image_crop_name = 'test_crop' + time.strftime('_%Y-%m-%d_%H-%M-%S') + '.jpg'
        cv2.imwrite(upload_folder + image_crop_name, image_crop)
        image_name = image_crop_name

        check_color_bounding_box = request.form['check_dominant']

    image_path_name = os.path.join(location_image_web, image_name)
    loaded_image = image.load_img(image_path_name, target_size=(512, 512))

    most_similar = visual_recommender.get_top_n(
        models=model_list,
        img=loaded_image,
        number_recommend=recommends_per_model,
        distance_method=selected_distance
    )

    info = {}
    for model_string in most_similar.keys():
        if config.get('settings', 'dataset_source') == 'elastic':
            info[model_string] = [loader.dataset[i]['image_name'] for i in most_similar[model_string]]
        else:
            info[model_string] = [loader.dataset[i] for i in most_similar[model_string]]
        if 'color' in selected_value_dominant or check_color_bounding_box == 'true':
            message = build_message(identity, None, f"Dominant Color selected.")
            logger.debug(json.dumps(message))
            a = time.time()
            img = skimage.io.imread(upload_folder + image_name)
            images_rgb = [loader.dataset[i]['dominant_color_rgb'] for i in most_similar[model_string]]
            sort_by_color = color_reranker.rank(img, images_rgb, info[model_string])
            message = build_message(identity, None,
                                    f'Dominant Color Rerank takes {(time.time() - a):.3f} seconds')
            logger.debug(json.dumps(message))
            info[model_string] = sort_by_color
            message = build_message(identity, None, f"sort_by_color: {sort_by_color}")
            logger.debug(json.dumps(message))
        if 'texture' in selected_value_dominant:
            gray_image = get_gray_image(upload_folder + image_name)
            gray_images = []
            for image_index in most_similar[model_string]:
                image_path = glob.glob(dataset_folder + "/**/" + str(loader.dataset[image_index]['image_name']),
                                       recursive=True)
                gray_images.append(get_gray_image(image_path[0]))
            message = build_message(identity, None, f"Dominant Texture selected.")
            logger.debug(json.dumps(message))
            _, sort_by_texture = rank_images_texture(gray_image, gray_images, info[model_string])
            info[model_string] = sort_by_texture
            message = build_message(identity, None, f"sort_by_texture: {sort_by_texture}")
            logger.debug(json.dumps(message))

    return render_template('master_page.html', filename=filename, info=info, image_detect=detected_objects,
                           models=model_list, image_shape=image_shape)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/displays/<filename>')
def display_images(filename):
    category_name = glob.glob(dataset_folder + "/**/" + filename, recursive=True)
    filename = urlparse(category_name[0])[2].replace('static', '')

    identity, request_time_string = initiate_request(request)
    message = build_message(identity, None, f"image url: {url_for('static', filename=filename)}")
    logger.debug(json.dumps(message))
    return redirect(url_for('static', filename=filename), code=301)


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', port='5002', threaded=False)

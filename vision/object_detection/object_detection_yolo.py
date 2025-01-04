import cv2
import numpy as np


def load_image(img_path):
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (500, 500))  # , None, fx=0.4, fy=0.4
    height, width, channels = img.shape
    return img, height, width, channels


def display_blob(blob):
    for b in blob:
        for n, image_blob in enumerate(b):
            cv2.imshow(str(n), image_blob)


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    box = []
    percentage = []
    ids_class = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.1:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                box.append([x, y, w, h])
                percentage.append(float(conf))
                ids_class.append(class_id)
    return box, percentage, ids_class


def draw_labels(box, score, colors, id_class, classes, image):
    # Set the NMS Threshold and the IoU threshold
    indexes = cv2.dnn.NMSBoxes(box, score, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(box)):
        if i in indexes:
            x, y, w, h = box[i]
            label = str(classes[id_class[i]])
            color = colors[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 5), font, 1, color, 1)
    # cv2.imshow("Image", image)


def load_model(model_weights, configure, label_names):
    net = cv2.dnn.readNet(model_weights, configure)
    with open(label_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


base_dir = "models/object_detection/yolo-openimage-600/"
model_weights = base_dir + "yolov3-openimages.weights"
configure = base_dir + "yolov3-openimages.cfg"
label_names = base_dir + "openimages.names"
# base_dir = "models/object_detection/yolo_custom_dataset/yolo-obj-random_0_size_608/"
# model_weights = base_dir + "yolo-obj_last.weights"
# configure = base_dir + "yolo-obj.cfg"
# label_names = base_dir + "obj.names"


class YoloV3ObjectDetector:
    def __init__(self):
        self.model, self.classes, self.colors, self.output_layers = load_model(model_weights, configure, label_names)

    def detect(self, image_path):
        image, height, width, channels = load_image(image_path)
        blob, outputs = detect_objects(image, self.model, self.output_layers)
        box, percentage, ids_class = get_box_dimensions(outputs, height, width)
        # print(f'boxes: {box}\npercentages: {percentage}\nclass_ids: {ids_class}')
        draw_labels(box, percentage, self.colors, ids_class, self.classes, image)
        image_shape = image.shape

        return box, percentage, ids_class, image_shape

    def get_names(self, class_ids):
        return [open(label_names, "r").readlines()[i] for i in class_ids]

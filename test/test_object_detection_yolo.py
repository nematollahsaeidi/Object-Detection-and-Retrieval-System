
def test_images():
    import os
    from vision.object_detection.object_detection_yolo import YoloV3ObjectDetector
    DIRECTORY = './test/test_images/'

    location_image = os.path.join(DIRECTORY, 'category-men-sport-t-shirts-polos_116981190.jpg')
    boxes, percentages, class_ids, image_main_shape = YoloV3ObjectDetector().detect(location_image)
    assert class_ids == [68, 432]
    assert boxes == [[-62, -144, 2067, 2819], [57, 88, 1939, 2547]]

    location_image = os.path.join(DIRECTORY, 'category-men-tracksuits-sets_105636158.jpg')
    boxes, percentages, class_ids, image_main_shape = YoloV3ObjectDetector().detect(location_image)
    assert class_ids == [432, 432]
    assert boxes == [[35, 185, 1095, 1638], [1246, 456, 650, 1975]]
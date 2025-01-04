
def test_image_color():
    from vision.filters.color.dominant_color_hist import dominant_color_histogram
    DIRECTORY = './test/test_images/'

    dominant_color = dominant_color_histogram('category-men-sport-t-shirts-polos_116981190.jpg', DIRECTORY)
    assert dominant_color == (61, 61, 61)

    dominant_color = dominant_color_histogram('category-men-tracksuits-sets_105636158.jpg', DIRECTORY)
    assert dominant_color == (245, 245, 245)


def test_image_name_color():
    from vision.filters.color.dominant_color_hist import dominant_color_histogram
    from vision.filters.color.webcolor import get_color_name
    DIRECTORY = './test/test_images/'

    dominant_color = dominant_color_histogram('category-men-sport-t-shirts-polos_116981190.jpg', DIRECTORY)
    assert get_color_name(dominant_color)[1] == 'darkslategray'

    dominant_color = dominant_color_histogram('category-men-tracksuits-sets_105636158.jpg', DIRECTORY)
    assert get_color_name(dominant_color)[1] == 'whitesmoke'
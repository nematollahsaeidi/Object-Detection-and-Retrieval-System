from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImageGenerator:
    def __init__(self):
        self.image_generation_initiate = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            vertical_flip=False,
            brightness_range=(0.4, 1.5),
            # zca_whitening=False,
            # zca_epsilon=1e-06,
            # featurewise_center=False,
            # samplewise_center=False,
            # featurewise_std_normalization=False,
            # samplewise_std_normalization=False,
            # rescale=1. / 255
        )

    def image_generation(self, directory_folders, save_image_dir, number_all_images, augmented_per_image=5):
        augmented_generation = self.image_generation_initiate.flow_from_directory(directory=directory_folders,
                                                                                  save_to_dir=save_image_dir,
                                                                                  batch_size=1,
                                                                                  class_mode='categorical',
                                                                                  save_prefix='image',
                                                                                  save_format='jpg')
        data_gen_key = list(augmented_generation.class_indices.keys())
        for i in range(number_all_images):
            augmented_images = [augmented_generation[i][0][0] for K in range(augmented_per_image)]


directory = 'category'
save_to_dir = 'category_augmented'
number_per_image = 4
n_all_images = 7
IG = ImageGenerator()
IG.image_generation(directory, save_to_dir, n_all_images, number_per_image)

from imageai.Prediction import ImagePrediction

import os
import numpy as np


class Prediction:
    def __init__(self):
        execution_path_model = 'object_detection_localization/I%mageAI-master'

        self.prediction = ImagePrediction()
        self.prediction.setModelTypeAsInceptionV3()
        inception_path = os.path.join(execution_path_model, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
        self.prediction.setModelPath(inception_path)
        self.prediction.loadModel()
        # define arrays according to imageai's categories and digistyle's categories
        self.dict_array1 = {'category-women-sport-t-shirts-and-polos': 1, 'category-men-ties': 2,
                            'category-men-formal-shoes': 3, 'category-kids-trousers-jumpsuits': 1,
                            'category-casual-shoes-for-women': 3, 'category-boys-suits': 1,
                            'category-men-tracksuits-sets': 1, 'sales': 0,
                            'category-men-sport-shoes-': 3, 'category-women-blouse': 1,
                            'category-women-coat-': 1, 'category-girls-scarves': 1,
                            'category-girls-skirts': 1, 'category-men-backpacks': 6,
                            'category-women-gold-watch-pendant': 9,
                            'category-women-sport-sweatshirts-and-hoodies': 1,
                            'category-boys-shoe-care-and-accessories': 10,
                            'category-men-sweatshirts': 1, 'category-women-sport-shorts': 1,
                            'category-women-shoe-care-accessories': 10,
                            'category-girls-flat-shoes': 3, 'category-men-sport-accessories': 10,
                            'category-kids-shorts': 1, 'category-girls-sandals': 3,
                            'mens-apparel-shop': 0, 'category-kids-skirts': 1,
                            'category-women-headwear': 8, 'category-kids-footwear': 3,
                            'category-men-silver-jewelry': 9, 'category-men-tee-shirts-and-polos': 1,
                            'category-men-homewear': 1, 'category-men-ankle-boots': 3,
                            'category-kids-knitwear': 1, 'category-men-sport-trousers-jumpsuits': 1,
                            'category-women-skirts': 1, 'category-women-sport-tops': 1,
                            'category-women-homewear': 1, 'category-women-sport-shoes-': 3,
                            'category-women-flat-shoes': 3, 'category-women-tracksuits-and-sets': 1,
                            'category-girls-tops': 1, 'category-men-wallets-bags': 6,
                            'category-men-gifts-sets': 0, 'womens-apparel-shop': 0,
                            'category-women-eyewear': 9, 'category-women-sport-accessories': 10,
                            'category-women-ankle-boots': 3, 'category-kids-kids-suits': 1,
                            'category-men-shoe-care-accessories': 10, 'category-men-belts': 10,
                            'category-women-tee-shirts-and-polos': 1, 'category-women-belts': 10,
                            'category-boys-formal-shoes': 3, 'category-women-tops': 1,
                            'category-boys-backpacks': 6, 'category-girls-boots': 3,
                            'category-boys-headwear': 8, 'category-girls-footwear': 3,
                            'category-women-heeled-shoes': 3, 'category-men-sportswear': 1,
                            'category-girls-tee-shirts-polos': 1, 'category-women-ring': 9,
                            'category-baby-accessories': 10, 'category-kids-napkin-and-apron': 10,
                            'category-men-gloves-mittens': 10, 'category-digistyle-gift-card': 0,
                            'category-women-necklace': 9, 'category-boys-trousers-jumpsuits': 1,
                            'category-girls-sport-shoes': 3, 'category-boys-sweatshirts': 1,
                            'category-kids-accessories': 10, 'category-boys-socks-tights': 3,  # 11
                            'category-kids-tee-shirts-polos': 1, 'category-boys-accessories': 10,
                            'category-girls-accessories': 10, 'category-girls-dresses': 1,
                            'category-kids-headwear': 8, 'category-women-earrings': 9,
                            'category-men-outwear': 1, 'category-girls-headwear': 8,
                            'category-men-socks-tights': 3, 'category-women-jeans': 1,  # 11
                            'category-men-shoes': 3, 'category-women-scarves': 10,
                            'category-girls-shorts': 1, 'category-kids-sport-shoes-': 3,
                            'category-kids-shirts-': 1, 'category-women-socks-': 3,  # 11
                            'category-women-gloves-and-mittens': 10, 'category-girls-sweatshirts': 1,
                            'category-women-sport-skirt': 1, 'category-men-slippers': 3,
                            'category-women-shorts-': 1, 'category-men-boots': 3,
                            'category-women-blazers-and-suits': 1,
                            'category-women-hair-accessories': 10, 'category-boys-knitwear': 1,
                            'kids-apparel-shop': 0, 'category-men-homewear-bottom': 1,
                            'category-casual-shoes-for-men': 3,
                            'category-women-trousers-and-jumpsuits': 1, 'category-men-watches': 7,
                            'category-women-sport-outwear': 1, 'category-men-shawl': 1,
                            'category-women-ties': 2, 'category-manteau-and-poncho': 1,
                            'category-women-dresses': 1, 'category-men-bags': 6,
                            'category-kids-dresses': 1, 'category-women-boots': 3,
                            'category-women-suits-and-sets': 1, 'category-men-clothing': 1,
                            'category-girls-girls-suits': 1,
                            'category-women-keyring-and-keychain': 9,
                            'category-women-sport-trousers-and-jumpsuits': 1,
                            'category-women-accessories': 10, 'category-men-sport-shorts': 1,
                            'category-women-light-jacket': 1, 'category-women-gold-pendants': 9,
                            'category-men-sport-outwear': 1,
                            'category-women-wallets-and-cosmetic-bags': 6, 'category-men-jeans': 1,
                            'category-girls-clothes': 1, 'category-women-shoes': 3,
                            'category-girls-hair-accessories': 10, 'category-men-sandals': 3,
                            'category-women-bags': 6, 'category-women-knitwear': 1,
                            'category-girls-trousers-jumpsuits': 1, 'category-boys-clothes': 1,
                            'category-boys-shorts': 1, 'category-women-watches': 7,
                            'category-men-underwear': 1, 'category-men-sport-t-shirts-polos': 1,
                            'category-women-shirt': 1, 'category-boys-boots': 3,
                            'category-men-headwear': 8, 'category-women-clothing': 1,
                            'category-men-glass-accessories': 10, 'category-men-accessories': 10,
                            'category-men-outwear-vest': 1, 'category-boys-sport-shoes': 3,
                            'category-women-backpacks': 6, 'category-girls-knitwear': 1,
                            'category-women-gold-chain': 9, 'category-kids-gloves-and-mittens': 10,
                            'category-men-knitwear': 1, 'category-boys-tee-shirts-polos': 1,
                            'category-boys-shirts': 1, 'category-men-coat': 1,
                            'category-men-sport-sweatshirts-hoodies': 1, 'category-women-sandals': 3,
                            'category-women-sweatshirts': 1, 'category-men-eyewear': 10,
                            'category-men-warm-jacket': 1, 'category-men-shirts': 1,
                            'category-women-underwear': 1, 'category-women-silver-jewelry': 9,
                            'category-women-slippers': 3, 'category-women-gold-necklace': 9,
                            'category-girls-ankle-boots': 3, 'category-women-bracelet': 9,
                            'category-kids-sweatshirts': 1, 'category-boys-ankle-boots': 3,
                            'category-men-blazers-suits': 1, 'category-women-flip-flops': 3,
                            'category-women-warm-jacket': 1, 'category-women-jacket-': 1}

        # shirtandsuits=1, tie=2, shoes=3, bag=6, watch:7, head:8, jewelry:9, accessories:10, socks:11
        self.dict_array2 = {
            7: ['analog_clock', 'stopwatch', 'digital_watch', 'wall_clock', 'magnetic_compass', 'digital_clock',
                'barometer', 'sundial', 'bell_cote', 'stopwatch', 'cellular_telephone', 'hand-held_computer',
                'remote_control', 'gyromitra', 'dial_telephone', 'hand-held_computer', 'ocarina'],
            9: ['chain', 'necklace', 'whistle', 'knot', 'safety_pin', 'pick', 'swing', 'padlock', 'chain_mail',
                'neck_brace', 'rubber_eraser', 'matchstick', 'panpipe', 'thimble'],
            3: ['running_shoe', 'Muzzle', 'sandal', 'shoe_shop', 'clog', 'Loafer', 'cowboy_boot', 'sock',
                'christmas_stocking', 'dumbbell', 'barbell'],
            8: ['pickelhaube', 'cowboy_hat', 'sombrero', 'crash_helmet', 'shower_cap', 'bathing_cap', 'revolver',
                'sun_hat'],
            6: ['backpack', 'mailbag', 'sleeping_bag', 'purse', 'punching_bag', 'wallet', 'pencil_box',
                'bulletproof_vest'],
            1: ['sweatshirt', 'cardigan', 'jersey', 'abaya', 'suit', 'sweatshirt', 'gown', 'boxer', 'hoopskirt',
                'trench_coat', 'cloak', 'poncho', 'fur_coat', 'stole', 'lab_coat', 'swab', 'miniskirt',
                'maillot', 'Torch', 'apron', 'breastplate', 'bib', 'jean', 'sarong', 'pajama', 'miniskirt',
                'overskirt', 'swimming_trunks'],
            10: ['apron', 'diaper', 'mitten', 'lens_cap', 'buckle', 'knot', 'hook', 'sunglasses', 'sunglass'],
            # 11: ['sock', 'christmas_stocking', 'dumbbell', 'barbell'],
            2: ['windsor_tie', 'bow_tie', 'knot', 'seat_belt']}

    def prediction_recommend(self, recommend, execution_path_image, name_image):
        """
        -usage in other files:
        from prediction_similar_search import prediction_recommend
        recommend = prediction_recommend(recommend, location_image, name_image=name_image)
        """
        predictions, probabilities = self.prediction.predictImage(os.path.join(execution_path_image, name_image),
                                                                  result_count=1)
        for eachPrediction, eachProbability in zip(predictions, probabilities):
            print(eachPrediction, " : ", eachProbability)

        # extract category corresponding to digistyle dataset
        category_image_imageai = [w for w, v in self.dict_array2.items() if predictions[0] in v]

        # extract category corresponding to digistyle dataset
        category_image_dataset = [w for w, v in self.dict_array1.items() for i in category_image_imageai if
                                  i in np.array(v)]

        image_name_list = [x for x in recommend if x.split('_')[0] in category_image_dataset]
        return image_name_list, predictions, probabilities

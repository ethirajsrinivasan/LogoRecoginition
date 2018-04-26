import os

__all__ = ('CLASS_NAME', 'CNN_IN_WIDTH', 'CNN_IN_HEIGHT', 'CNN_IN_CH',
           'CNN_SHAPE', 'TRAIN_DIR', 'TRAIN_IMAGE_DIR',
           'CROPPED_AUG_IMAGE_DIR', 'ANNOT_FILE', 'ANNOT_FILE_WITH_BG')

CLASS_NAME = [
    'Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari',
    'Ford', 'Google', 'HP', 'Heineken', 'Intel', 'McDonalds', 'Mini', 'Nbc',
    'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks',
    'Texaco', 'Unicef', 'Vodafone', 'Yahoo', 'Background'
]

CNN_IN_WIDTH = 64
CNN_IN_HEIGHT = 32
CNN_IN_CH = 3
CNN_SHAPE = (CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH)

TRAIN_DIR = '/home/viki/Desktop/DataWarehousing/project/flickr_logos_27_dataset'
TRAIN_IMAGE_DIR = os.path.join(TRAIN_DIR, 'flickr_logos_27_dataset_images')
CROPPED_AUG_IMAGE_DIR = os.path.join(
    TRAIN_DIR, 'flickr_logos_27_dataset_cropped_augmented_images')
ANNOT_FILE = os.path.join(
    TRAIN_DIR, 'flickr_logos_27_dataset_training_set_annotation.txt')
ANNOT_FILE_WITH_BG = os.path.join(TRAIN_DIR, 'train_annot_with_bg_class.txt')
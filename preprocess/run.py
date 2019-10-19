import os
import glob
import random
import cv2 as cv
import shutil
import logging
import numpy as np
from preprocess.augment_functions import shrink_images, vertical_flip, change_hsv

logging.basicConfig()


def load_matched_path(image_path="data/original_annotated/input", mask_path="data/original_annotated/output"):
    match_dataset = list()

    rgb_img_list = glob.glob('{}/*.jpg'.format(image_path))
    mask_list = glob.glob('{}/*.png'.format(mask_path))

    for file in range(len(rgb_img_list)):
        current_mask = mask_list[file].split('\\')[1].split('.')[0]
        current_rgb = rgb_img_list[file].split('\\')[1].split('.')[0]

        if current_mask == current_rgb:
            match_dataset.append(
                ("{}/{}.jpg".format(image_path, current_rgb), "{}/{}.png".format(mask_path, current_mask)))

    return match_dataset


def initialize_augment_pipeline(save_dir='data/augmented'):
    logging.info("Begin to load images")
    data_set = load_matched_path()

    for _val in data_set:
        matching_img = cv.imread(_val[0])
        matching_mask = cv.imread(_val[1])

        logging.info("Resizing ---> {}".format(_val))
        resized_img, resized_mask = shrink_images(matching_img, matching_mask)

        logging.info("Saving Original --->")
        cv.imwrite('{}/{}'.format(save_dir, os.path.basename(_val[0])), resized_img)
        cv.imwrite('{}/{}'.format(save_dir, os.path.basename(_val[1])), resized_mask)

        logging.info("Filtering HSV ---> {}".format(_val))
        filtered_dataset = change_hsv(resized_img, resized_mask)

        for filter_key, filter_val in enumerate(filtered_dataset):
            logging.info("Flipping data set ---> {}_{}".format(_val, filter_key))
            flipped_img, flipped_mask = vertical_flip(filter_val[0], filter_val[1])
            basename = os.path.basename(_val[0]).split('.')[0]

            img_augmented_name = '{}/{}_{}.jpg'.format(save_dir, basename, filter_key)
            mask_augmented_name = '{}/{}_{}.png'.format(save_dir, basename, filter_key)

            logging.info("Saving ---> {} and {}".format(mask_augmented_name, img_augmented_name))
            cv.imwrite(img_augmented_name, flipped_img)
            cv.imwrite(mask_augmented_name, flipped_mask)


def split_rate(masked_images, split_amt):
    return int(len(masked_images) * (split_amt / 100))


def split_into_dir(data_list, move_to_path):
    for file in data_list:
        try:
            shutil.move(file[0], move_to_path)
            shutil.move(file[1], move_to_path)
        except Exception:
            logging.error('Failed.', exc_info=True)


def train_val_test_split(train_split=60, val_split=20, read_from='data/augmented', save_to='data/dataset'):
    all_images = glob.glob('{}/*.jpg'.format(read_from))
    all_masks = glob.glob('{}/*.png'.format(read_from))

    masked_images = list(zip(all_images, all_masks))
    random.shuffle(masked_images)

    train_split_rate = split_rate(masked_images, train_split)
    validation_split_rate = split_rate(masked_images, val_split)

    train_list = masked_images[:train_split_rate]
    val_list = masked_images[train_split_rate:train_split_rate + validation_split_rate]
    test_list = masked_images[train_split_rate + val_split:]

    split_into_dir(train_list, "{}/train_set".format(save_to))
    split_into_dir(val_list, "{}/validation_set".format(save_to))
    split_into_dir(test_list, "{}/test_set".format(save_to))

    logging.info("Data is successfully split")


def load_data_to_numpy(data_path='data/dataset/train_set', file_ext='jpg'):
    avail_data = glob.glob('{}/*.{}'.format(data_path, file_ext))
    return np.array([cv.imread(item) for item in avail_data])


if __name__ == '__main__':
    initialize_augment_pipeline()
    train_val_test_split()

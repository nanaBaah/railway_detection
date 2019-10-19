import numpy as np
import os
import cv2 as cv
import glob
import logging

from keras_preprocessing.image import ImageDataGenerator

all_images = []
image_paths = "data/input"

print("Begin loading of images ... ")

for image in glob.glob('{}/*.jpg'.format(image_paths)):
    all_images.append(cv.imread(image))

all_images = np.array(all_images)

print("All images are loaded successfully ... ")

print("Augmentation process begins ...")

# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     rescale=0.5,
                     samplewise_center=True,
                     featurewise_std_normalization=True,
                     horizontal_flip=True)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(all_images, augment=True, seed=seed)
# mask_datagen.fit(masks, augment=True, seed=seed)

print("*************")

image_gen = image_datagen.flow(all_images, save_to_dir="C:/Users/nana/PycharmProjects/Lane_detection/data/augmented/")

aug = np.array([next(image_gen).astype(np.uint8) for i in range(1)])

image_generator = image_datagen.flow_from_directory(
    "data/input/",
    target_size=(960, 540),
    class_mode=None,
    seed=seed,
    save_to_dir="data/augmented/")

# mask_generator = mask_datagen.flow_from_directory(
#     'data/masks',
#     class_mode=None,
#     seed=seed)

# combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)

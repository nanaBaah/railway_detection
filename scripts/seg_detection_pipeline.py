from datetime import datetime

import cv2 as cv
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from preprocess.run import load_data_to_numpy
from scripts.segmentation_model import create_model

HEIGHT = 270
WIDTH = 480
CHANNEL = 3

batch_size = 32
epochs = 10
pool_size = (2, 2)
input_shape = (HEIGHT, WIDTH, CHANNEL)

X_train = load_data_to_numpy(data_path='data/dataset/train_set/train_images/images', file_ext='jpg')
y_train = load_data_to_numpy(data_path='data/dataset/train_set/train_images/masks', file_ext='png')

X_val = load_data_to_numpy(data_path='data/dataset/validation_set/train_images/val_images', file_ext='jpg')
y_val = load_data_to_numpy(data_path='data/dataset/validation_set/train_images/val_masks', file_ext='png')

X_test = load_data_to_numpy(data_path='data/dataset/test_set/test_images/images', file_ext='jpg')
y_test = load_data_to_numpy(data_path='data/dataset/test_set/test_masks/masks', file_ext='png')

data_gen_args = dict(rescale=1. / 255)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1

train_image_generator = image_datagen.flow_from_directory('data/dataset/train_set/train_images',
                                                          class_mode=None,
                                                          target_size=(HEIGHT, WIDTH),
                                                          seed=seed,
                                                          batch_size=32)
train_mask_generator = mask_datagen.flow_from_directory('data/dataset/train_set/train_masks',
                                                        class_mode=None,
                                                        target_size=(HEIGHT, WIDTH),
                                                        seed=seed,
                                                        batch_size=32)

train_generator = zip(train_image_generator, train_mask_generator)

val_image_generator = image_datagen.flow_from_directory('data/dataset/train_set/train_images',
                                                        class_mode=None,
                                                        target_size=(HEIGHT, WIDTH),
                                                        seed=seed,
                                                        batch_size=32)
val_mask_generator = mask_datagen.flow_from_directory('data/dataset/train_set/train_masks',
                                                      class_mode=None,
                                                      target_size=(HEIGHT, WIDTH),
                                                      seed=seed,
                                                      batch_size=32)

validation_generator = zip(val_image_generator, val_mask_generator)

model = create_model(input_shape=input_shape, pool_size=pool_size)

early_stop = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=1000,
    verbose=1,
    restore_best_weights=True
)

file_path = 'weights/weights.{epoch:02d}-{val_loss:.6f}.hdf'

checkpoint = ModelCheckpoint(
    file_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='auto',
    period=1
)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensor_board_callback = keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=200,
    verbose=2,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[checkpoint, early_stop, tensor_board_callback]
)

# Test

test_img = cv.resize(X_test[0], (WIDTH, HEIGHT), interpolation=cv.INTER_LINEAR)
test_img = np.array(test_img * (1.0 / 255)).astype(np.float32)
test_img = np.expand_dims(test_img, axis=0)

plt.figure()
plt.imshow(test_img[0])
plt.show()

y_hat = model.predict(test_img, verbose=1, steps=20)

test_mask = np.array(y_hat[0] * 255).astype(np.uint8)

plt.figure()
plt.imshow(test_mask + test_img[0])
plt.show()


model.save('seg_model.h5')
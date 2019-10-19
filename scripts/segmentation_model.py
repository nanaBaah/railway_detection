import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers import Dropout, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


def create_model(input_shape, pool_size):

    # Here is the actual neural network ###
    model = Sequential()
    # Normalizes incoming inputs. First layer needs the input shape to work
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv1'))
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv2'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv3'))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv4'))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv5'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv6'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv7'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=pool_size))

    # Upsample 1
    model.add(UpSampling2D(size=pool_size))

    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv1'))
    model.add(Dropout(0.2))

    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv2'))
    model.add(Dropout(0.2))

    # Upsample 2
    model.add(UpSampling2D(size=pool_size))

    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv3'))
    model.add(Dropout(0.2))

    model.add(Conv2DTranspose(32, (4, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv4'))
    model.add(Dropout(0.2))

    model.add(Conv2DTranspose(16, (4, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv5'))
    model.add(Dropout(0.2))

    # Upsample 3
    model.add(UpSampling2D(size=pool_size))
    # Deconv 6
    model.add(Conv2DTranspose(16, (4, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv6'))
    # Final layer - only including one channel so 1 filter
    model.add(Conv2DTranspose(3, (4, 3), padding='valid', strides=(1, 1), activation='relu', name='Final'))

    # End of network ###

    # Compiling and training the model
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.summary()

    return model

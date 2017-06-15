from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math

def generator_model():
    layers = [
        Dense(1024, input_dim=100),
        Activation('tanh'),
        Dense(7 * 7 * 128),
        BatchNormalization(),
        Activation('tanh'),
        Reshape((7, 7, 128), input_shape=(7 * 7 * 128,)),
        UpSampling2D(size=(2, 2)),  # 14x14
        Conv2D(64, (5, 5), padding='same'),
        Activation('tanh'),
        UpSampling2D(size=(2, 2)),  # 28x28
        Conv2D(1, (5, 5), padding='same'),
        Activation('tanh')        
    ]
    model = Sequential(layers)
    return model

def discriminator_model():
    layers = [
        Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)),
        Activation('tanh'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (5, 5)),
        Activation('tanh'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(1024),
        Activation('tanh'),
        Dense(1),
        Activation('sigmoid')
    ]
    model = Sequential(layers)
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) *shape[0], j * shape[1]:(j + 1) * shape[1]] = img[:, :, 0]
    return image

def train(batch_size):
    # load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    generator.compile(loss='binary_crossentropy', optimizer='SGD')
    
    # generator: trainable, discriminator: freeze
    discriminator_on_generator.summary()
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)

    # discriminator: trainable
    discriminator.trainable = True
    discriminator.summary()
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    noise = np.zeros((batch_size, 100))
    
    for epoch in range(20):
        print('epoch:', epoch)
        num_batches = int(X_train.shape[0] / batch_size)
        print('number of batches', num_batches)
        for index in range(num_batches):
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]

            generated_images = generator.predict(noise, verbose=0)
            
            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            
            d_loss = discriminator.train_on_batch(X, y)
            
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, 100)

            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * batch_size)

            print('epoch: %d, batch: %d, g_loss: %f, d_loss: %f' % (epoch, index, g_loss, d_loss))

        image = combine_images(generated_images)
        image = image * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save('epoch-%03d.png' % epoch)

train(batch_size=128)

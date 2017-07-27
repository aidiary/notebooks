import numpy as np
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.applications.vgg19 import VGG19
from keras.callbacks import CSVLogger, ModelCheckpoint

IMG_WIDTH, IMG_HEIGHT = 368, 368

vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

for i in range(8):
    vgg19.layers.pop()
vgg19.outputs = [vgg19.layers[-1].output]

m = Sequential()
m.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3_CPM',
             input_shape=vgg19.layers[-1].output_shape[1:]))
m.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv4_CPM'))

# stage1 part confidence map
m.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1_CMP_L2'))
m.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv2_CMP_L2'))
m.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv3_CMP_L2'))
m.add(Conv2D(512, (1, 1), activation='relu', padding='same', name='block5_conv4_CMP_L2'))
m.add(Conv2D(1,   (1, 1), activation='relu', padding='same', name='block5_conv5_CMP_L2'))
m.add(Reshape((46 * 46 * 1, )))

# PoseNet
model = Model(inputs=vgg19.inputs, outputs=m(vgg19.outputs))
model.summary()

data = np.load('data/input_img_data.npy')
heatmaps = np.load('data/heatmaps.npy')
filenames = np.load('data/filenames.npy')

heatmaps = heatmaps.reshape((-1, 46 * 46 * 1))

logger = CSVLogger('train.log')
weight_file = 'train.{epoch:02d}-{loss:.3f}.h5'
checkpoint = ModelCheckpoint(
    weight_file,
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='auto')

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(data, heatmaps, batch_size=16, epochs=20, verbose=1, callbacks=[logger, checkpoint])

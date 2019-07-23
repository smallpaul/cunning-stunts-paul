from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.enable_eager_execution()

# for rtx 20xx cards
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import loadData

AUTOTUNE = tf.data.experimental.AUTOTUNE
LABEL_COUNT = 1108
BATCH_SIZE = 10
BUFFER_SIZE = 1
IMAGE_SIZE = 256


def preprocess_image(path, label):
    image = tf.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0
    return image, label


def make_tf_dataset(x, y):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(preprocess_image)
    ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=BUFFER_SIZE))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


x_train, y_train = loadData.get_unsequenced_data()
x_test, y_test = loadData.get_unsequenced_test()
label_names = [str(label) for label in range(LABEL_COUNT)]

train_ds = make_tf_dataset(x_train, y_train)
test_ds = make_tf_dataset(x_test, y_test)

# model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True,
#                                                                     weights=None,
#                                                                     input_tensor=None,
#                                                                     input_shape=(512, 512, 1),
#                                                                     pooling='avg',
#                                                                     classes=LABEL_COUNT)

model = tf.keras.applications.mobilenet.MobileNet(include_top=True,
                                                  weights=None,
                                                  input_tensor=None,
                                                  input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                                                  classes=LABEL_COUNT)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [tf.keras.callbacks.ModelCheckpoint('./mdoel.h5', verbose=1),
             tf.keras.callbacks.TensorBoard()]

model.fit(train_ds,
          epochs=200,
          steps_per_epoch=int(len(x_train)/BATCH_SIZE),
          validation_data=test_ds,
          validation_steps=int(len(x_test)/BATCH_SIZE),
          callbacks=callbacks)
# model.evaluate(test_ds)

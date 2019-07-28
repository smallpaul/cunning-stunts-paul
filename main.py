from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf

import loadData

tf.enable_eager_execution()

# for rtx 20xx cards
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

LABEL_COUNT = 1108
BATCH_SIZE = 20
BUFFER_SIZE = 1
IMAGE_SIZE = 256


def pre_process_image(path, label):
    image = tf.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image /= 255.0
    return image, label


def make_tf_data_set(x, y):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(pre_process_image)
    ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=BUFFER_SIZE))
    ds = ds.batch(BATCH_SIZE)
    return ds


x_train, y_train = loadData.get_unsequenced_data()
x_test, y_test = loadData.get_unsequenced_test()
label_names = [str(label) for label in range(LABEL_COUNT)]

train_ds = make_tf_data_set(x_train, y_train)
test_ds = make_tf_data_set(x_test, y_test)

model_path = 'models/myModel'
new_net = False
if os.path.isfile(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = tf.keras.applications.xception.Xception(include_top=True,
                                                    weights=None,
                                                    input_tensor=None,
                                                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                                                    classes=LABEL_COUNT)

    # model = tf.keras.applications.mobilenet.MobileNet(include_top=True,
    #                                                   weights=None,
    #                                                   input_tensor=None,
    #                                                   input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
    #                                                   classes=LABEL_COUNT)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #model = tf.keras.Model(input=model.inputs, output=model.layers[5].output) # for chopping off the end to put in recurrent
    new_net = True

accuracy = 0
if not new_net:
    result = model.evaluate(test_ds,
                            steps=int(len(x_test)/BATCH_SIZE))
    accuracy = result[1]


class SaveModel(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global model, accuracy
        if logs is None:
            print('no logs :(')
            return
        val_acc = float(logs.get('val_acc'))
        if accuracy < val_acc:
            print('Saving network')
            if os.path.isfile(model_path):
                tf.keras.models.save_model(
                    model,
                    model_path,
                    overwrite=True,
                    include_optimizer=True)
            else:
                tf.keras.models.save_model(
                    model,
                    model_path,
                    overwrite=True,
                    include_optimizer=True)
            accuracy = val_acc


callbacks = [tf.keras.callbacks.TensorBoard(update_freq='batch'),
             SaveModel()]

model.fit(train_ds,
          epochs=20,
          steps_per_epoch=int(len(x_train)/BATCH_SIZE),
          validation_data=test_ds,
          validation_steps=int(len(x_test)/BATCH_SIZE),
          callbacks=callbacks)


import os
import numpy as np # linear algebra
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

height, width = 150, 150

def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255.0, label

def augment_img(image, label):
    global height, width
    image = tf.image.resize_with_crop_or_pad(image, height+6, width+6)
    image = tf.image.random_crop(image, size=[height, width, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2)
    image = tf.image.random_saturation(image, 0, 2)
    return image, label


if __name__ == '__main__':
    (cars_train, cars_test), cars_info = tfds.load(
        "cars196",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    print(cars_info)
    # print(tfds.list_builders())
    cars_train = cars_train.map(lambda x, y: (tf.image.resize(x, (height, width)), y))
    # validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    cars_test = cars_test.map(lambda x, y: (tf.image.resize(x, (height, width)), y))

    # cars_train = keras.utils.normalize(cars_train)
    # cars_test = keras.utils.normalize(cars_test)

    # cars_train = cars_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    cars_train = cars_train.cache().map(augment_img)
    # cache saves training data to memory, prefetch 64 batches
    cars_train = cars_train.shuffle(cars_info.splits["train"].num_examples).batch(32).prefetch(10)

    # cars_test = cars_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    cars_test = cars_test.cache().map(augment_img).batch(32).prefetch(10)

    print("Train ", cars_info.splits["train"].num_examples)
    print("Test ", cars_info.splits["test"].num_examples)

    # data_augmentation = keras.Sequential(
    #     [
    #         # layers.experimental.preprocessing.Resizing(height=64, width=64),
    #         layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
    #         layers.experimental.preprocessing.RandomContrast(factor=0.1),
    #         layers.experimental.preprocessing.RandomCrop(height=32, width=32),
    #     ]
    # )

    # data_augmentation = keras.Sequential(
    #     [
    #         tf.image
    #     ]
    # )


    # model = keras.Sequential(
    #     [
    #         keras.Input((150, 150, 3)),
    #         # data_augmentation,
    #         layers.experimental.preprocessing.Normalization(),
    #         layers.Conv2D(32, 9, activation='relu'),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Conv2D(64, 3, activation='relu'),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Conv2D(64, 9, activation='relu'),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Conv2D(128, 9, activation='relu'),
    #         layers.Flatten(),
    #         layers.Dropout(rate=0.2),
    #         layers.Dense(196),
    #     ]
    # )

    # model = keras.Sequential(
    #     [
    #         keras.Input((150, 150, 3)),
    #         # data_augmentation,
    #         layers.experimental.preprocessing.Normalization(),
    #         layers.Conv2D(32, 3, activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Conv2D(64, 3, activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Conv2D(64, 9, activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Conv2D(128, 9, activation="relu"),
    #         layers.Flatten(),
    #         layers.Dropout(rate=0.2),
    #         layers.Dense(196, activation="softmax"),
    #     ]
    # )
    #
    # print(model.summary())
    #
    # model.compile(
    #     optimizer=keras.optimizers.Adam(lr=0.001),
    #     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=["accuracy"],
    # )
    #
    # model.fit(cars_train, epochs=5, verbose=2)
    # model.evaluate(cars_test)

    base_model = tf.keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(height, width, 3),
        include_top=False,  # Do not include the final ImageNet classifier layer at the top.
    )

    base_model.trainable = False  # We want to update all the model weights, so set this to true.

    # Create new model on surrounding our pretrained base model.
    inputs = tf.keras.Input(shape=(height, width, 3))

    # Pre-trained Xception weights requires that input be normalized
    # from (0, 255) to a range (-1., +1.), the normalization layer
    # does the following, outputs = (inputs - mean) / sqrt(var)
    mean = np.array([127.5] * 3)
    norm_layer = keras.layers.experimental.preprocessing.Normalization(mean=mean, variance=mean ** 2)
    # Scale inputs to [-1, +1]
    x = norm_layer(inputs)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(
        x)  # this is a neural network operation to help adapt the features learned by the pretrained model to our specific task.
    x = keras.layers.Dropout(0.5)(x)  # Regularize with dropout
    num_outputs = cars_info.features[
        'label'].num_classes  # This is the number of output variables we want, 196 in this case.
    outputs = keras.layers.Dense(num_outputs, activation="softmax")(
        x)  # Use activation=softmax for classification, and activation=None for regression.
    model = keras.Model(inputs, outputs)

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])

    epochs = 100
    model.fit(cars_train, epochs=epochs, validation_data=cars_test)


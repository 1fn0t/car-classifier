
import os
import numpy as np # linear algebra
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

from keras import backend as K

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

def f1_score(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(K.round(y_pred), 'float32')
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1_score = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1_score

# class PrecisionRecallF1Score(Metric):
#     def __init__(self, **kwargs):
#         super(PrecisionRecallF1Score, self).__init__(**kwargs)
#         self.precision = self.add_weight(name='precision', initializer='zeros')
#         self.recall = self.add_weight(name='recall', initializer='zeros')
#         self.f1_score = self.add_weight(name='f1_score', initializer='zeros')
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = K.cast(y_true, y_pred.dtype)
#         tp = K.sum(y_true * y_pred)
#         fp = K.sum((1 - y_true) * y_pred)
#         fn = K.sum(y_true * (1 - y_pred))
#
#         precision = tp / (tp + fp + K.epsilon())
#         recall = tp / (tp + fn + K.epsilon())
#         f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
#         self.precision.assign_add(precision)
#         self.recall.assign_add(recall)
#         self.f1_score.assign_add(f1_score)
#
#     def result(self):
#         return {'precision': self.precision,
#                 'recall': self.recall,
#                 'f1_score': self.f1_score}
#
#     def reset_states(self):
#         self.precision.assign(0)
#         self.recall.assign(0)
#         self.f1_score.assign(0)


if __name__ == '__main__':

    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    (cars_train, cars_test), cars_info = tfds.load(
        "cars196",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    print(cars_info)
    cars_train = cars_train.map(lambda x, y: (tf.image.resize(x, (height, width)), y))
    cars_test = cars_test.map(lambda x, y: (tf.image.resize(x, (height, width)), y))


    # cars_train = cars_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    cars_train = cars_train.cache().map(augment_img)
    # cache saves training data to memory, prefetch 64 batches
    cars_train = cars_train.shuffle(cars_info.splits["train"].num_examples).batch(32).prefetch(10)

    # cars_test = cars_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    cars_test = cars_test.cache().map(augment_img).batch(32).prefetch(10)

    print("Train ", cars_info.splits["train"].num_examples)
    print("Test ", cars_info.splits["test"].num_examples)

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
                  metrics=['accuracy', f1_score])

    epochs = 100
    model.fit(cars_train, epochs=epochs, validation_data=cars_test)


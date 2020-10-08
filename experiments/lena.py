import PIL
import pathlib
import numpy as np
import tensorflow as tf


def load_lena():
    data_dir = tf.keras.utils.get_file(
        origin=('https://upload.wikimedia.org/wikipedia/en/7/7d/'
                'Lenna_%28test_image%29.png'),
        fname='Lena')
    data_dir = pathlib.Path(data_dir)
    x = PIL.Image.open(str(data_dir))
    return tf.constant(np.asarray(x), dtype='int32')


def binary_repr(x):
    num_bits = (
        tf.cast(
            tf.math.log(
                tf.cast(
                    tf.reduce_max(x),
                    'float32')),
            'int32')
        + 1)
    x = tf.cast(x, 'int32')
    return tf.math.mod(
        tf.bitwise.right_shift(
            x[..., tf.newaxis],
            tf.range(num_bits)),
        2)
    
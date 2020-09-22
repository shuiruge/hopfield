import tensorflow as tf

__all__ = (
    'sign',
    'tempered',
    'softly',
    'softly_binarize',
    'SoftBinarization',
    'softly_argmax',
)


def sign(x):
    """Return -1 if x > 0 else 1 element-wisely, with dtype conserved."""
    y = tf.where(x > 0, 1, -1)
    y = tf.cast(y, x.dtype)
    return y


def tempered(T, fn):
    """Converts f(x) to f(x/T).

    Parameters
    ----------
    T : float
        The temperature.
    fn : callable

    Returns
    -------
    callable
    """
    T = float(T)
    return lambda x: fn(x / T)


def softly(fn):
    r"""Decorator that returns func(*x, **kwargs), with the gradients on the x
    :math:`\partial f_i / \partial x_j = \delta_{i j}`, i.e. an unit Jacobian,
    or say, an identity vector-Jacobian-product.
    """

    def identity(*dy):
        if len(dy) == 1:
            dy = dy[0]
        return dy

    @tf.custom_gradient
    def softly_fn(*args, **kwargs):
        y = fn(*args, **kwargs)
        return y, identity

    return softly_fn


def softly_binarize(x, threshold, minval=0, maxval=1, from_logits=False):
    r"""Returns `maxval` if x > threshold else `minval`, element-wisely, with
    the gradients :math:`\partial f_i / \partial x_j = \delta_{i j}`, i.e. an
    unit Jacobian.

    Parameters
    ----------
    x : tensor
    threshold : float
    minval : real number
    maxval : real number
    from_logits : bool, optional
        If true, then softly binarize sigmoid(x) instead of x.

    Returns
    -------
    tensor
        The same shape and dtype as x.
    """

    @softly
    def binarize(x):
        y = tf.where(x > threshold, maxval, minval)
        y = tf.cast(y, x.dtype)
        return y

    return binarize(tf.nn.sigmoid(x)) if from_logits else binarize(x)


class SoftBinarization(tf.keras.layers.Layer):
    """For using in tf.keras.Sequential.

    If in training phase, then do nothing. Otherwise, make soft binarization.

    Parameters
    ----------
    threshold : float
    from_logits : bool, optional
        If true, then softly binarize sigmoid(x) instead of x.
    """

    def __init__(self, threshold, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.from_logits = from_logits

    def get_config(self):
        config = super().get_config()
        config['threshold'] = self.threshold
        config['from_logits'] = self.from_logits
        return config

    def call(self, x, training=None):
        if self.from_logits:
            x = tf.nn.sigmoid(x)
        if training:
            return x
        return softly_binarize(x, self.threshold)


# TODO: test this function.
def softly_argmax(x, axis, from_logits=False):
    r"""Returns 1 if the element x[..., i, ...] == max(x[..., i, ...]), along
    axis i, with the gradients
    :math:`\partial f_i / \partial x_j = \delta_{i j}`, i.e. an unit Jacobian.

    Parameters
    ----------
    x : tensor
    axis : int
    from_logits : bool, optional
        If true, then softly argmax softmax(x) instead of x.

    Returns
    -------
    tensor
        The same shape and dtype as x.
    """

    @softly
    def argmax(x):
        max_x = tf.reduce_max(x, axis=axis, keepdims=True)
        y = tf.where(x == max_x, 1, 0)
        y = tf.cast(y, x.dtype)
        return y

    return argmax(tf.nn.softmax(x)) if from_logits else argmax(x)
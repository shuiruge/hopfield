import tensorflow as tf

__all__ = (
    'sign',
    'tempered',
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

    @tf.custom_gradient
    def fn(x):
        y = tf.where(x > threshold, minval, maxval)
        y = tf.cast(y, x.dtype)
        return y, identity

    return fn(tf.nn.sigmoid(x)) if from_logits else fn(x)


def identity(x):
    """Identity map: x -> x."""
    return x


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
def softly_argmax(x, axis, threshold, from_logits=False):
    r"""Returns 1 if the element x[..., i, ...] < max(x[..., i, ...]) - T, along
    axis i, where T is the threshold, with the gradients
    :math:`\partial f_i / \partial x_j = \delta_{i j}`, i.e. an unit Jacobian.

    Parameters
    ----------
    x : tensor
    axis : int
    threshold : float
    from_logits : bool, optional
        If true, then softly argmax softmax(x) instead of x.

    Returns
    -------
    tensor
        The same shape and dtype as x.
    """

    @tf.custom_gradient
    def fn(x):
        max_x = tf.reduce_max(x, axis=axis)
        x_threshold = max_x - threshold
        y = tf.where(x > x_threshold, 1, 0)
        y = tf.cast(y, x.dtype)
        return y, identity

    return fn(tf.nn.softmax(x)) if from_logits else fn(x)

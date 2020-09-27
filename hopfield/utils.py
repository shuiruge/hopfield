import tensorflow as tf

__all__ = (
    'step',
    'softstep',
    'softsign',
    'softbinary',
    'tempered',
)


def step(x, threshold, minval, maxval):
    """Returns maxval if x > threshold else minval, element-wisely.

    Parameters
    ----------
    x : tensor
    threshold : float or tensor
    minval : float or tensor
    maxval : float or tensor

    Returns
    -------
    tensor
    """
    y = tf.where(x > threshold, maxval, minval)
    y = tf.cast(y, x.dtype)
    return y


def softstep(x, threshold, minval, maxval, T):
    """Returns maxval if x > threshold else minval, element-wisely.
    The gradient is replaced by a soft version, with "temperature" T.

    Parameters
    ----------
    x : tensor
    threshold : float or tensor
    minval : float or tensor
    maxval : float or tensor
    T : float, optional

    Returns
    -------
    tensor
    """

    @tf.custom_gradient
    def fn(x):
        z = tf.nn.sigmoid((x - threshold) / T)

        def grad_fn(dy):
            return dy * (maxval - minval) * z * (1 - z) / T

        y = step(x, threshold, minval, maxval)
        return y, grad_fn

    return fn(x)


def softsign(x, T=1e-0):
    """
    Parameters
    ----------
    x : tensor
    T : float, optional

    Returns
    -------
    tensor
        The same shape and dtype as x.
    """
    return softstep(x, threshold=0, minval=-1, maxval=1, T=T)


def softbinary(x, T=1e-0):
    """
    Parameters
    ----------
    x : tensor
    T : float, optional

    Returns
    -------
    tensor
        The same shape and dtype as x.
    """
    return softstep(x, threshold=0, minval=0, maxval=1, T=T)


def tempered(T, fn=None):
    """Converts f(x) to f(x/T).

    Parameters
    ----------
    T : float
        The temperature.
    fn : callable, optional
        If `None`, returns a decorator.

    Returns
    -------
    callable or decorator
    """
    T = float(T)

    def decorator(fn):
        return lambda x: fn(x / T)

    return decorator if fn is None else decorator(fn)

import tensorflow as tf

__all__ = (
    'soft_step',
    'soft_sign',
    'tempered',
)


def step(x, x0, minval, maxval):
    y = tf.where(x > x0, maxval, minval)
    y = tf.cast(y, x.dtype)
    return y


def soft_step(x, x0, minval, maxval, T=1e-0):
    """Returns maxval if x > x0 else minval, element-wisely. The gradient
    is replaced by a soft version, with "temperature" T.

    Parameters
    ----------
    x : tensor
    x0 : float or tensor
    minval : float or tensor
    maxval : float or tensor
    T : float, optional

    Returns
    -------
    tensor
    """
    @tf.custom_gradient
    def fn(x):
        z = tf.nn.sigmoid((x - x0) / T)

        def grad_fn(dy):
            return dy * (maxval - minval) * z * (1 - z) / T

        y = step(x, x0, minval, maxval)
        return y, grad_fn

    return fn(x)


def soft_sign(x):
    """
    Parameters
    ----------
    x : tensor

    Returns
    -------
    tensor
        The same shape and dtype as x.
    """
    return soft_step(x, x0=0, minval=-1, maxval=1)


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

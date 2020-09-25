"""Collections of re-constructors."""

import numpy as np
import tensorflow as tf
from .utils import soft_sign

__all__ = (
    'NonidentityRecon',
    'DenseRecon',
    'ModernDenseRecon',
    'Conv2dRecon',
)


class NonidentityRecon(tf.keras.layers.Layer):
    """Base class of re-constructor which is further constrainted
    to avoid learning to be an identity map."""


class DenseRecon(NonidentityRecon):
    """Fully connected non-identity re-constructor.

    Parameters
    ----------
    activation : callable, optional
    """

    def __init__(self,
                 activation=soft_sign,
                 **kwargs):
        super().__init__(**kwargs)
        self.activation = activation

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        return config

    def build(self, input_shape):
        depth = input_shape[-1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=[depth, depth],
            initializer="glorot_uniform",
            constraint=symmetrize_and_mask_diagonal,
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=[depth],
            initializer="zeros",
            trainable=True)
        super().build(input_shape)

    def call(self, x):
        f = self.activation
        W, b = self.kernel, self.bias
        return f(x @ W + b)


def symmetrize_and_mask_diagonal(kernel):
    """Symmetric kernel with vanishing diagonal.

    Parameters
    ----------
    kernel : tensor
        Shape (N, N) for a positive integer N.

    Returns
    -------
    tensor
        The shape and dtype as the input.
    """
    w = (kernel + tf.transpose(kernel)) / 2
    w = tf.linalg.set_diag(w, tf.zeros(kernel.shape[0:-1]))
    return w


class ModernDenseRecon(NonidentityRecon):
    """
    References
    ----------
    1. https://ml-jku.github.io/hopfield-layers/

    Parameters
    ----------
    activation : callable, optional
    n : int, optional
        Number of patterns to learn. If `None`, then `memory` shall be
        provided.
    memory : array-like, optional
        Shape `[n, depth]` where `n` is the same as the argument `n`,
        and `depth` is the dimension of inputs. If `None`, then `n`
        shall be provided.
    """

    def __init__(self,
                 activation=soft_sign,
                 n=None,
                 memory=None,
                 **kwargs):
        super().__init__(**kwargs)

        if n is None and memory is None:
            raise ValueError('Either `n` or `memory` has to be provided.')

        self.activation = activation
        self.n = n
        self.memory = memory

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        config['n'] = self.n
        config['memory'] = self.memory
        return config

    def build(self, input_shape):
        depth = input_shape[-1]
        if self.memory is None:
            n = self.n
            initializer = 'glorot_uniform'
        else:
            n = self.memory.shape[0]

            def initializer(_, dtype=None):
                return tf.constant(self.memory, dtype=dtype)

        self.kernel = self.add_weight(name='kernel',
                                      shape=[n, depth],
                                      initializer=initializer,
                                      trainable=True)
        super().build(input_shape)

    def call(self, x):
        f = self.activation
        W = self.kernel
        return f(x @ tf.transpose(W)) @ W


class Conv2dRecon(NonidentityRecon):
    """Cellular automata based non-identity re-constructor.

    References
    ----------
    1. Cellular automata as convolutional neural networks (arXiv: 1809.02942).

    Parameters
    ----------
    filters : int
    kernel_size : int
    activation : callable, optional
    binarize: callable, optional
        Binarization method for non-training process. If `None`, then no
        binarization.
    flatten : bool, optional
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 activation=soft_sign,
                 flatten=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.activation = activation
        self.flatten = flatten

    def get_config(self):
        config = super().get_config()
        config['filters'] = self.filters
        config['kernel_size'] = self.kernel_size
        config['activation'] = self.activation
        config['flatten'] = self.flatten
        return config

    def build(self, input_shape):
        recon_layers = [
            tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                activation='relu',
                padding='same',
                kernel_constraint=mask_center,
            ),
            tf.keras.layers.Conv2D(1, 1, activation=self.activation),
        ]
        if self.flatten:
            depth = input_shape[-1]
            two_dim_shape = [int(np.sqrt(depth))] * 2 + [1]
            recon_layers.insert(0, tf.keras.layers.Reshape(two_dim_shape))
            recon_layers.append(tf.keras.layers.Reshape([depth]))
        self._recon = tf.keras.Sequential(recon_layers)
        self._recon.build(input_shape)

        super().build(input_shape)

    def call(self, x):
        return self._recon(x)


def _get_center_mask(dim: int) -> np.array:
    assert dim % 2 == 1
    center = int(dim / 2) + 1
    mask = np.ones([dim, dim, 1, 1])
    mask[center, center, 0, 0] = 0
    return mask


def mask_center(kernel):
    # kernel shape: [dim, dim, n_channels, n_filters]
    dim, *_ = kernel.get_shape().as_list()
    mask = tf.constant(_get_center_mask(dim), dtype=kernel.dtype)
    return kernel * mask
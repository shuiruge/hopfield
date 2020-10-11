"""Collections of re-constructors."""

import numpy as np
import tensorflow as tf
from .utils import softsign, step

__all__ = (
    'NonidentityRecon',
    'get_recon_loss',
    'DenseRecon',
    'ModernDenseRecon',
    'Conv2dRecon',
    'RBMRecon',
    'HebbianRBMRecon',
)


class NonidentityRecon(tf.keras.layers.Layer):
    """Base class of re-constructor which is further constrainted
    to avoid learning to be an identity map.

    This class provides a method for computing the re-construction loass
    `get_recon_loss`, mapping the input tensor and re-constructed tensor
    to a scalar. Defaults to MAE. You can override this method for your
    need.
    """

    @staticmethod
    def get_recon_loss(x, recon_x):
        return tf.reduce_mean(tf.abs(x - recon_x))


def get_recon_loss(non_identity_recon, x, recon_x):
    """Returns the re-construction loss defined by the non-identity re-
    constructor `non_identity_recon`, with input `x` and the re-constructed
    `recon_x`.

    Parameters
    ----------
    non_identity_recon : NonidentityRecon
    x : tensor
    recon_x : tensor

    Returns
    -------
    scalar
    """
    return non_identity_recon.get_recon_loss(x, recon_x)


def sign(x):
    y = tf.where(x > 0.5, 1, -1)
    return tf.cast(y, x.dtype)


# XXX: Using `softsign` as activation will not gain the promised properties
#      of Hopfield network, but using tanh with sign binarization re-gains.
#      This is not consistent with the discussion in `RMBRecon`.
class DenseRecon(NonidentityRecon):
    """Fully connected non-identity re-constructor.

    Parameters
    ----------
    activation : callable
        The activation can be soft, for continuous-state Hopfield networks.
        Or soften-hard, for discrete-state Hopfield networks. In the later,
        the softness means that, even though the output is hard, the gradient
        exists (via custom gradient).
    """

    def __init__(self,
                 activation=tf.math.tanh,
                 binarize=sign,
                 **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.binarize = binarize

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

    def call(self, x, training=None):
        f = self.activation
        W, b = self.kernel, self.bias
        y = f(x @ W + b)
        if not training:
            y = self.binarize(y)
        return y


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
    activation : callable
        The activation can be soft, for continuous-state Hopfield networks.
        Or soften-hard, for discrete-state Hopfield networks. In the later,
        the softness means that, even though the output is hard, the gradient
        exists (via custom gradient). However, discrete-state Hopfield networks
        is suggested for using this re-constructor.
    n : int, optional
        Number of patterns to learn. If `None`, then `memory` shall be
        provided.
    memory : array-like, optional
        Shape `[n, depth]` where `n` is the same as the argument `n`,
        and `depth` is the dimension of inputs. If `None`, then `n`
        shall be provided.
    """

    def __init__(self,
                 activation,
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
    activation : callable
        The activation can be soft, for continuous-state Hopfield networks.
        Or soften-hard, for discrete-state Hopfield networks. In the later,
        the softness means that, even though the output is hard, the gradient
        exists (via custom gradient).
    flatten : bool, optional
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 activation=softsign,
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


class RBMRecon(NonidentityRecon):
    """Restricted Boltzmann machine (RBM) based non-identity re-constructor.

    RBM is a kind of Hopfield network, with a block anti-diagonal weight-matrix
    and async updation.

    RBM -- LDPC -- AE

    Notes
    -----
    Non-identity:
        The latent dimension `latent_dim` shall be smaller the ambient
        dimension, for ensuring the non-identity of the re-constructor.

    References
    ----------
    * Introduction to low-density parity-check (LDPC) code:
        1. https://medium.com/5g-nr/ldpc-low-density-parity-check-code-8a4444153934
    * Introduction to Boltzmann machine:
        2. https://medium.com/edureka/restricted-boltzmann-machine-tutorial-991ae688c154
    * Relation between Boltzmann machine and auto-encoder:
        3. https://www.cs.cmu.edu/~rsalakhu/talk_Simons_part2_pdf.pdf

    Parameters
    ----------
    latent_dim : int
    softsign : callable, optional
        Soft version of `sign` function. Softness means that the output of the
        function is hard version, while the gradient is smooth (i.e. custom
        gradient).
    T : float, optional
        The "temperature" characterizing how "soft" the gradient is.
    use_bias : bool, optional
        If `False`, then all the biases are set to zeros and non-trainable.
    """

    def __init__(self,
                 latent_dim,
                 softsign=softsign,
                 T=1e-0,
                 use_bias=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.softsign = softsign
        self.T = float(T)
        self.use_bias = use_bias

    def get_config(self):
        config = super().get_config()
        config['latent_dim'] = self.latent_dim
        config['softsign'] = self.softsign
        config['T'] = self.T
        config['use_bias'] = self.use_bias
        return config

    def build(self, input_shape):
        depth = input_shape[-1]
        assert depth > self.latent_dim

        self.kernel = self.add_weight(
            name='kernel',
            shape=[depth, self.latent_dim],
            initializer='glorot_uniform',
            trainable=True)
        self.latent_bias = self.add_weight(
            name='latent_bias',
            shape=[self.latent_dim],
            initializer='zeros',
            trainable=self.use_bias)
        self.ambient_bias = self.add_weight(
            name='ambient_bias',
            shape=[depth],
            initializer='zeros',
            trainable=self.use_bias)
        super().build(input_shape)

    def call(self, x):
        W, b, v = self.kernel, self.latent_bias, self.ambient_bias

        def f(x):
            return self.softsign(x / self.T)

        z = f(x @ W + b)
        y = f(z @ tf.transpose(W) + v)
        return y


# TODO
class HebbianRBMRecon(NonidentityRecon):

    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def get_config(self):
        config = super().get_config()
        config['latent_dim'] = self.latent_dim
        config['softsign'] = self.softsign
        return config

    def build(self, input_shape):
        depth = input_shape[-1]
        assert depth > self.latent_dim

        self.kernel = self.add_weight(
            name='kernel',
            shape=[depth, self.latent_dim],
            initializer=lambda shape, dtype: tf.cast(
                step(tf.random.uniform(shape), 0.5, 1, -1),
                dtype),
            trainable=False)
        self.latent_bias = self.add_weight(
            name='latent_bias',
            shape=[self.latent_dim],
            initializer='zeros',
            trainable=False)
        self.ambient_bias = self.add_weight(
            name='ambient_bias',
            shape=[depth],
            initializer='zeros',
            trainable=False)
        super().build(input_shape)

    def call(self, x, training=None):

        def sign(x):
            return step(x, 0, -1, 1)

        W, b, v = self.kernel, self.latent_bias, self.ambient_bias
        x1 = x
        z1 = sign(x1 @ W + b)
        x2 = sign(z1 @ tf.transpose(W) + v)

        if training:

            def outer_prod(x, y):
                return x[..., :, tf.newaxis] * y[..., tf.newaxis, :]

            z2 = sign(x2 @ W + b)
            dW = outer_prod(x1, z1) - outer_prod(x2, z2)
            dW = tf.reduce_sum(dW, axis=0)
            self.kernel.assign_add(dW)

            db = tf.reduce_sum(z1 - z2, axis=0)
            self.latent_bias.assign_add(db)

            dv = tf.reduce_sum(x1 - x2, axis=0)
            self.ambient_bias.assign_add(dv)

        return x2

    @staticmethod
    def get_recon_loss(x, recon_x):
        # re-construction loss is useless, since no SGD to do.
        return 0.

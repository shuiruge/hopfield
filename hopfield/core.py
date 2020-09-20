import tensorflow as tf

__all__ = (
    'DiscreteTimeHopfieldLayer',
)


class DiscreteTimeHopfieldLayer(tf.keras.layers.Layer):
    """
    References
    ----------
    1. Information Theory, Inference, and Learning Algorithm (D. Mackay),
       chapter 42.

    Parameters
    ----------
    non_identity_recon : NonidentityRecon
    max_steps : int
    async_ratio : float, optional
        Percentage of "bits" to be randomly masked in each updation.
    relax_tol : float, optional
    reg_factor : float, optional

    Attributes
    ----------
    final_step : int32 scalar
    """

    def __init__(self,
                 non_identity_recon,
                 max_steps: int,
                 async_ratio=0.,
                 relax_tol=1e-3,
                 reg_factor=1e-0,
                 **kwargs):
        super().__init__(**kwargs)
        self.non_identity_recon = non_identity_recon
        self.max_steps = max_steps
        self.async_ratio = float(async_ratio)
        self.relax_tol = float(relax_tol)
        self.reg_factor = float(reg_factor)

        self.final_step = tf.Variable(0, trainable=False)

    def get_config(self):
        config = super().get_config()
        config['non_identity_recon'] = self.non_identity_recon
        config['max_steps'] = self.max_steps
        config['async_ratio'] = self.async_ratio
        config['relax_tol'] = self.relax_tol
        config['reg_factor'] = self.reg_factor
        return config

    def _learn(self, x):
        r = self.non_identity_recon(x, training=True)
        loss = tf.reduce_mean(tf.abs(x - r))
        return r, loss

    def _update(self, x):
        final_step = 0
        for step in tf.range(self.max_steps):
            next_x = self._update_step(x)
            if diff(next_x, x) < self.relax_tol:
                final_step = step
                break
            x = next_x
        else:
            final_step = self.max_steps
        return x, final_step

    def _update_step(self, x):
        y = self.non_identity_recon(x, training=False)
        if self.async_ratio > 0:
            # mask has no batch dim
            mask = tf.where(
                tf.random.uniform(y.shape[1:]) < self.async_ratio,
                0., 1.)
            y *= mask[tf.newaxis, ...]
        return y

    def call(self, x, training=None):
        if training:
            y, loss = self._learn(x)
            self.add_loss(self.reg_factor * loss)
        else:
            y, final_step = self._update(x)
            self.final_step.assign(final_step)
        return y


def diff(x, y):
    return tf.reduce_max(tf.abs(x - y))

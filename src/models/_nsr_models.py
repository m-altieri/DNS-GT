import tensorflow as tf
import yaml


# UNUSED
class NSRModel(tf.keras.Model):
    def __init__(self, H, D, **conf):
        super(NSRModel, self).__init__()

        self.H = H
        self.D = D

        _default_conf_file = "../conf/NSRModel_conf.yaml"
        with open(_default_conf_file, "r") as f:
            _default_conf = yaml.safe_load(f)
        conf = _default_conf | conf  # override default with passed arguments

        # don't use .get(), it's better if it raises an exception
        # if smth is missing
        self._host_dim = conf["host_dim"]
        self._domain_dim = conf["domain_dim"]
        self._host_code_dim = conf["host_code_dim"]
        self._domain_code_dim = conf["domain_code_dim"]

        self.nsr_hosts = tf.Variable(
            tf.random.uniform((H, self._host_dim)),
            trainable=False,
            name="NSR Hosts",
            dtype=tf.dtypes.float32,
        )

        self.nsr_domains = tf.Variable(
            tf.random.uniform((D, self._domain_dim)),
            trainable=False,
            name="NSR Domains",
            dtype=tf.dtypes.float32,
        )

        self.nsr = (self.nsr_hosts, self.nsr_domains)

        # @TODO adj_hosts and adj_domains must be replaced with the actual
        # neighborhoods (decide how to calculate them)
        self.adj_hosts = tf.Variable(
            tf.ones((H, H)), trainable=False, name="Adj Hosts", dtype=tf.dtypes.float32
        )

        self.adj_domains = tf.Variable(
            tf.ones((D, D)),
            trainable=False,
            name="Adj Domains",
            dtype=tf.dtypes.float32,
        )

        self.updater = _NSRUpdater()
        self.autoencoder = _Autoencoder()

    # Fader component (for now a normal function, it can
    # be made trainable)
    @tf.function
    def _fader(gamma_h=0.9, gamma_d=0.9):
        self.nsr_hosts = tf.math.multiply(self.nsr_hosts, gamma_h)
        self.nsr_domains = tf.math.multiply(self.nsr_domains, gamma_d)

    def call(self, inputs, training=False):
        self._fader()
        self.nsr_hosts, self.nsr_domains = self.updater(inputs)
        rec = self.autoencoder(self.nsr_hosts, self.nsr_domains)

        return rec

    # @TODO i think i have to to the fading phase in train_step,
    # because it's the only way to respect the "x = y" approach
    # (
    def train_step(self, data):
        x, y = data

        self._fader()
        with tf.GradientTape() as tape:
            self.nsr = self.updater(inputs)
            rec = self.autoencoder(self.nsr)
            # loss = compute_loss(self.nsr, rec)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(self.nsr, rec)
        return {m.name: m.result() for m in self.metrics}


# Updater component
class _NSRUpdater(tf.keras.layers.Layer):
    def __init__(self, host_dim, domain_dim):
        super(_NSRUpdater, self).__init__()

        self.host_updater = tf.keras.layers.Dense(host_dim)
        self.domain_updater = tf.keras.layers.Dense(domain_dim)

    def call(self, inputs):
        # [[H, hdim], [D, ddim], [B, 2]]
        nsr_hosts, nsr_domains, queries = inputs
        pass


# Autoencoder component
class _Autoencoder(tf.keras.layers.Layer):
    def __init__(self, host_code_dim, domain_code_dim):
        super(_Autoencoder, self).__init__()

        self.host_encoders = [_Encoder(host_code_dim) for h in range(H)]
        self.domain_encoders = [_Encoder(domain_code_dim) for d in range(D)]

        self.host_decoder = _Decoder(self._host_dim)
        self.domain_decoder = _Decoder(self._domain_dim)

    def call(self, inputs):
        codes = [self.host_encoders[h](inputs[:, h]) for h in range(self.H)]
        codes = tf.stack(codes, axis=1)
        reconstruction = self.host_decoder(codes)
        pass


class _Encoder(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(_Encoder, self).__init__()

        self.output_dim = output_dim

        self.layer = tf.keras.layers.Dense(output_dim, activation="relu")

    def call(self, inputs):
        return self.layer(inputs)


class _Decoder(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(_Decoder, self).__init__()

        self.layer = tf.keras.layers.Dense(output_dim, activation="relu")

    def call(self, inputs):
        return self.layer(inputs)

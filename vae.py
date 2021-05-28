import tensorflow as tf
from utils import load_mnist


class Vae(tf.keras.Model):
    def __init__(self) -> None:
        super(Vae, self).__init__()
        self.encoder_ = self.make_encoder()
        self.decoder_ = self.make_decoder()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.model_ = self.make_model()

        # self.decoder_(out_encoder)

        # self.model = tf.keras.Model(self.encoder_, self.decoder_)

    def make_model(self):
        ## Encoder
        encoder_input = tf.keras.layers.Input((28, 28))

        layer = tf.keras.layers.Flatten()(encoder_input)
        layer = tf.keras.layers.Dense(64, activation="tanh")(layer)
        layer = tf.keras.layers.Dense(32, activation="tanh")(layer)

        means = tf.keras.layers.Dense(2, activation="linear", name="means")(layer)
        logvar = tf.keras.layers.Dense(2, activation="linear", name="logvar")(layer)

        ## Decoder
        output_encoder = tf.keras.layers.concatenate([means, logvar])

        layer = tf.keras.layers.Dense(32, activation="tanh")(output_encoder)
        layer = tf.keras.layers.Dense(64, activation="tanh")(layer)
        decoded_layer = tf.keras.layers.Dense(784, activation="linear")(layer)

        return tf.keras.Model(encoder_input, [means, logvar, decoded_layer])

    def make_encoder(self):
        encoder_input = tf.keras.layers.Input((28, 28))

        layer = tf.keras.layers.Flatten()(encoder_input)
        layer = tf.keras.layers.Dense(64, activation="tanh")(layer)
        layer = tf.keras.layers.Dense(32, activation="tanh")(layer)

        means = tf.keras.layers.Dense(2, activation="linear", name="means")(layer)
        logvar = tf.keras.layers.Dense(2, activation="linear", name="logvar")(layer)

        return tf.keras.Model(encoder_input, [means, logvar])

    def make_decoder(self):

        decoder_input = tf.keras.layers.Input((2, 1))

        layer = tf.keras.layers.Dense(32, activation="tanh")(decoder_input)
        layer = tf.keras.layers.Dense(64, activation="tanh")(layer)
        last_layer = tf.keras.layers.Dense(784, activation="linear")(layer)

        return tf.keras.Model(decoder_input, last_layer)

    def encode_and_sample(self, x):
        (m, s, _) = self.encode(x)
        return tf.random.normal((2,), m, s)

    def fit(self, x, epochs, lr):
        pass

    def train_step(self, x, epochs, lr):
        with tf.GradientTape() as tape_encoder, tf.GradientTape() as tape_decoder:
            res = self.encode_and_sample(x)
            gradients = tape_encoder.gradient(res, self.encoder_.variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.encoder_.trainable_variables)
            )

    def encode(self, X):
        m, s, _ = self.model_(X)
        return (m, s)

    def decode(self, X):
        return self.decoder_(X)


if __name__ == "__main__":
    train_loader = load_mnist()
    model = Vae()
    # print(model.encoder_.summary())
    # tf.keras.utils.plot_model(model.encoder_, to_file="./encoder.png", show_shapes=True)
    # print(model.decoder_.summary())
    # tf.keras.utils.plot_model(model.decoder_, to_file="./decoder.png", show_shapes=True)

    print(model.model_.summary())
    tf.keras.utils.plot_model(model.model_, to_file="./model.png", show_shapes=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.util.nest import flatten
from utils import load_mnist2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import tqdm


class Gan:
    def __init__(
        self,
        X: np.ndarray,
        epochs: int,
        batch_size: int,
        activation,
        last_activation_encode=None,
        last_activation_decode=None,
    ):
        tf.random.set_seed(42)
        self.X = X
        self.epochs = epochs
        self.batch_size = batch_size
        # self.shape = np.shape(X)[1]

        self.activation = activation
        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_opt = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_opt = tf.keras.optimizers.Adam(1e-4)

    def make_generator(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation=self.activation),
                tf.keras.layers.Dense(28 * 28),
                tf.keras.layers.Reshape((28, 28, 1)),
            ]
        )
        return model

    def make_discriminator(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation=self.activation),
                tf.keras.layers.Dense(64, activation=self.activation),
                tf.keras.layers.Dense(32, activation=self.activation),
                tf.keras.layers.Dense(1),
            ]
        )
        return model

    def fit(self):
        for epoch in tqdm.tqdm(range(self.epochs)):
            for image_batch in self.X:
                self.train_step(image_batch)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def generator_loss(self, fake_output):
        loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        return loss

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, 100])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator.trainable_variables
            )
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables
            )

            self.generator_opt.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables)
            )
            self.discriminator_opt.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables)
            )


if __name__ == "__main__":
    X = load_mnist2()
    # size = np.shape(X)[1]

    model = Gan(X, 10, 100, "relu")
    model.fit()

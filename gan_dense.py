import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.models import Sequential
from utils import load_mnist2

class GanDense:
    def __init__(self, X, batch_size):
        self.X = X
        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # self.generator_opt = tf.keras.optimizers.Adam(1e-4)
        # self.discriminator_opt = tf.keras.optimizers.Adam(1e-4)

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)


    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def train_step(self, batch, noise_dim):
        noise = tf.random.normal([batch.shape[0], noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator[0](noise, training=True)

            real_output = self.discriminator[0](batch, training=True)
            fake_output = self.discriminator[0](generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator[0].trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator[0].trainable_variables)

        self.generator[1].apply_gradients(zip(gradients_of_generator, self.generator[0].trainable_variables))
        self.discriminator[1].apply_gradients(zip(gradients_of_discriminator, self.discriminator[0].trainable_variables))
        return gen_loss, disc_loss

    def fit(self, epochs, noise_dim):
        seed = tf.random.normal([16, noise_dim])
        checkpoint_dir = './dense_gan_models'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator[1], discriminator_optimizer=self.discriminator[1], generator=self.generator[0], discriminator=self.discriminator[0])

        for epoch in range(epochs):
            total_gen_loss = 0
            total_disc_loss = 0
            batch_count = len(self.X)
            for idx, batch in enumerate(self.X):
                print("\rbatches {} / {}".format(idx, batch_count), end=" ")
                gen_loss, disc_loss = self.train_step(batch, noise_dim)
                total_gen_loss += gen_loss
                total_disc_loss += disc_loss
            if (epoch + 1) % 20 == 0:
                self.generate_and_save_images(seed)
            print("\rgenerator loss {:0.4f} disc_loss {:0.4f}".format(total_gen_loss / batch_count,
                                                                        total_disc_loss / batch_count))

    def make_generator(self):
        model = Sequential([
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(28 * 28),
            tf.keras.layers.Reshape((28, 28, 1))
        ])


        optimizer = tf.keras.optimizers.Adam(1e-4)
        return model, optimizer


    def make_discriminator(self):
        model = Sequential([

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        optimizer = tf.keras.optimizers.Adam(1e-4)
        return model, optimizer

    def generate_and_save_images(self, noise_input, cmap="gray"):
        predictions = self.generator[0](noise_input, training=False)
        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((predictions[i] + 1) / 2.0, cmap=cmap)
            plt.axis('off')

        # plt.savefig(os.path.join(path, 'image_at_epoch_{:04d}.png'.format(epoch)))
        plt.show()

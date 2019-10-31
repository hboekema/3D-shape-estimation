""" Generate a GAN network """

import numpy as np
from datetime import datetime
from tqdm import tqdm
import collections

import keras
from keras.layers import Dense, Dropout, Input, Flatten, Reshape, BatchNormalization
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

from smpl_np import SMPLModel

np.random.seed(datetime.now().seconds)


class GAN:
    def __init__(self, disc_input_dim, gen_input_dim=(100,), example_data=None):
        self._gen_dim = gen_input_dim
        self._disc_dim = disc_input_dim

        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.gan = self.create_gan()

        # Data
        self.ex_data = example_data
        self.X = None
        self.X_val = None

    def D_loss(self, D_real, D_gen):
        """ Discriminator loss function"""
        return -np.mean(np.log(D_real) + np.log(1. - D_gen))

    def G_loss(self, D_gen):
        """ Generator loss function """
        return -np.mean(np.log(D_gen))

    def create_generator(self):
        """ Create an MLP network for the generator """
        # Use a sequential network
        generator = Sequential()

        # Take a noisy input and generate arbitrary parameters
        generator.add(Dense(units=256, input_dim=self._gen_dim))
        generator.add(BatchNormalization(momentum=0.99))
        generator.add(Dropout(0.3))

        generator.add(Dense(units=512))
        generator.add(BatchNormalization(momentum=0.99))
        generator.add(Dropout(0.3))

        generator.add(Dense(units=1024))
        generator.add(BatchNormalization(momentum=0.99))
        generator.add(Dropout(0.2))

        generator.add(Dense(units=128))
        generator.add(BatchNormalization(momentum=0.99))
        generator.add(Dropout(0.5))

        # Output the SMPL parameters in the correct dimensions
        generator.add(Dense(units=np.prod(self._disc_dim), activation="tanh"))
        generator.add(Reshape(self._disc_dim))

        # Compile the model with the Adam optimiser
        generator.compile(loss="binary_crossentropy", optimizer=Adam(1e-4, beta_1=0.5), metrics=["accuracy"])

        return generator

    def create_discriminator(self):
        """ Create an MLP network for the discriminator """
        # Use a sequential network
        discriminator = Sequential()

        # Take the SMPL parameters from the real models and synthesised models
        discriminator.add(Flatten())
        discriminator.add(Dense(units=1024, input_dim=np.prod(self._disc_dim)))
        discriminator.add(LeakyReLU())
        discriminator.add(Dropout(0.4))

        discriminator.add(Dense(units=512))
        discriminator.add(LeakyReLU())
        discriminator.add(Dropout(0.2))

        discriminator.add(Dense(units=256))
        discriminator.add(LeakyReLU())

        # The output layer is a score of how well the SMPL model matches real data
        discriminator.add(Dense(units=1, activation='sigmoid'))

        # Compile the model with the Adam optimiser
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(2e-4, beta_1=0.5), metrics=["accuracy"])

        return discriminator

    def create_gan(self):
        """ Create the generative adversarial network """
        # Specify the input and output of the network
        gan_input = Input(shape=(100,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)

        # Compile the model
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=Adam(1.5e-4, beta_1=0.5), metrics=["accuracy"])
        return gan

    def fit(self, X, X_val=None):
        """ Preprocess the data """
        np.random.shuffle(self.X)

        angles = X.sum(axis=1)
        X /= angles[:, np.newaxis]

        self.X = np.concatenate([angles, X], axis=1)

        if X_val is not None:
            np.random.shuffle(self.X_val)

            angles = X_val.sum(axis=1)
            X_val /= angles[:, np.newaxis]

            self.X_val = np.concatenate([angles, X_val], axis=1)

    def save_generated_smpl(self, epoch, examples=3, render=True):
        """ Save generated SMPL parameters """
        if self.ex_data is None:
            self.ex_data = np.random.randn(examples, self._gen_dim)
        generated_params = self.generator.predict(self.ex_data)
        for i, params in enumerate(generated_params):
            np.save("../gen_pose_params/gan_example.E{:03d}:{:02d}.npy".format(epoch, i), params)

        if render:
            smpl = SMPLModel('../model.pkl')

            beta = np.zeros(shape=(10,))
            trans = np.zeros(shape=(3,))

            for i, pose_params in enumerate(generated_params):
                smpl.set_params(beta=beta, pose=pose_params, trans=trans)
                smpl.save_to_obj('../meshes/smpl_np.E{:03d}:{:02d}.obj'.format(epoch, i))

    def train(self, epochs, steps_per_epoch, batch_size, epoch_save_period=10):
        """ Train the generator network """
        loss_dict = collections.OrderedDict()

        for epoch in range(1, epochs + 1):
            print("\nEpoch %d" % epoch)
            with tqdm(total=steps_per_epoch) as pbar:
                for i in range(steps_per_epoch):
                    # Generate random noise as an input to initialize the generator
                    noise = np.random.randn(batch_size, 100)

                    # Generate fake SMPL parameters from noisy input
                    generated_params = self.generator.predict(noise)

                    # Get a random set of real SMPL parameters and corrupt these with additive noise
                    param_batch = self.X[np.random.randint(low=0, high=self.X.shape[0], size=batch_size)]
                    param_batch += np.random.uniform(low=-0.02, high=0.02, size=param_batch.shape)

                    # Construct different batches of real and fake data
                    X = np.concatenate([param_batch, generated_params])

                    # Labels for real and generated data - soft labels help in training
                    y_dis = np.zeros(2 * batch_size)
                    y_dis[:batch_size] = np.random.uniform(0.9, 1.0, size=batch_size)
                    y_dis[batch_size:] = np.random.uniform(0.0, 0.1, size=batch_size)

                    # Pre-train discriminator on fake and real data before starting the GAN
                    self.discriminator.trainable = True
                    disc_outputs = self.discriminator.train_on_batch(X, y_dis)
                    loss_dict["disc_loss"] = "{:.03f}".format(disc_outputs[0])
                    loss_dict["disc_acc"] = "{:.03f}".format(disc_outputs[1])

                    # Treat the generated data as real
                    noise = np.random.randn(batch_size, 100)
                    y_gen = np.ones(batch_size)

                    # During the training of the GAN the weights of discriminator are fixed
                    self.discriminator.trainable = False

                    # Train the GAN by alternating the training of the Discriminator
                    # and training the chained GAN model with the Discriminator’s weights frozen
                    gan_outputs = self.gan.train_on_batch(noise, y_gen)
                    loss_dict["gan_loss"] = "{:.03f}".format(gan_outputs[0])
                    loss_dict["gan_acc"] = "{:.03f}".format(gan_outputs[1])

                    pbar.set_postfix(loss_dict, refresh=True)
                    pbar.update()

            tqdm.close(pbar)

            # Evaluate the model on an example batch
            fake_output = self.discriminator.predict(self.generator.predict(np.random.randn(batch_size, 100)))
            if self.X_val is not None:
                real_output = self.discriminator.predict(self.X_val[np.random.randint(
                    low=0, high=self.X_val.shape[0], size=batch_size)])
                print("Discriminator output on real data: {:04f}".format(np.mean(real_output)))
            print("Discriminator output on fake data: {:04f}".format(np.mean(fake_output)))

            if epoch == 1 or epoch % epoch_save_period == 0:
                self.save_generated_smpl(epoch, render=True)


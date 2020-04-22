
import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Concatenate, Lambda
from keras.activations import softplus, tanh
from keras.optimizers import Adam
from .agent import Agent


class Actor(Agent):
    """ Actor for the A2C Algorithm """
    def __init__(self, input_dim, output_dim, network, lr):
        Agent.__init__(self, input_dim, output_dim, lr, clipvalue=10)
        self.output_dim = self.output_dim[0]
        self.action_pl = K.placeholder(shape=(None, self.output_dim))
        self.advantages_pl = K.placeholder(shape=(None,))
        self.model = self.add_head(network)

        print(self.model.summary())

        # Count the number of trainable etc. weights
        #trainable_count = int(np.sum([K.count_params(w) for w in self.model.trainable_weights]))
        #non_trainable_count = int(np.sum([K.count_params(w) for w in self.model.non_trainable_weights]))

        #print('\nActor')
        #print('--------------------------------------------------')
        #print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        #print('Trainable params: {:,}'.format(trainable_count))
        #print('Non-trainable params: {:,}'.format(non_trainable_count))
        #print('--------------------------------------------------')

    def add_head(self, network):
        """ Add output head for the given input network """
        x = Dense(128, activation='relu', name="actor_head_slot")(network.output)

        # Get the distribution parameters
        #num_scale_elem = int((self.output_dim * (self.output_dim - 1))/2 + self.output_dim)
        #mu = Dense(self.output_dim, activation=lambda x: np.pi*tanh(x), name="pi_mu")(x)
        mu = Dense(self.output_dim, activation="linear", name="pi_mu")(x)
        print("mu shape: " + str(mu.shape))
        #scale_flat = Dense(num_scale_elem, activation="linear", name="scale_flat")(x)
        #scale_flat = Dense(self.output_dim * self.output_dim, activation="linear", name="scale_flat")(x)

        scale_diag = Dense(self.output_dim, activation="softplus", name="pi_scale_diag")(x)
        print("scale_diag shape: " + str(scale_diag.shape))
        #scale_off_diag = Dense(int((self.output_dim * (self.output_dim - 1))/2), activation="linear", name="pi_scale_off_diag")(x)
        #scale_flat = Concatenate()([scale_diag, scale_off_diag])
        #print(scale_flat)

        scale = Lambda(lambda x: K.tf.matrix_diag(x), name="pi_scale_unprotected")(scale_diag)
        scale = Lambda(lambda x: K.tf.contrib.distributions.matrix_diag_transform(x, transform=lambda x: x + 1e-5), name="pi_scale")(scale)
        print("scale shape: " + str(scale.shape))

        # Construct the scale matrix (defined as covariance = scale @ scale.T)
        #indices = list(zip(*np.tril_indices(self.output_dim)))
        #indices = [list(i) for i in indices]
        #scale = Lambda(lambda x: K.tf.sparse_to_dense(sparse_indices=indices, output_shape=[self.output_dim, self.output_dim], sparse_values=x, default_value=0, validate_indices=True))(scale_flat)
        #print(scale)
        #scale = Lambda(lambda x: K.tf.contrib.distributions.matrix_diag_transform(x, transform=softplus), name="pi_scale")(scale)
        #print(scale)

        # Return the (Normal) distribution parameters
        return Model(network.input, [mu, scale])

    def optimizer(self):
        """ Actor optimization uses Advantages + Entropy term to encourage exploration """
        # @TODO - check whether this works and is mathematically correct
        log_prob = self.log_prob(self.action_pl)
        eligibility = log_prob * K.stop_gradient(self.advantages_pl)
        entropy = self.entropy()      # entropy in bits
        loss = -K.mean(0.01 * entropy + eligibility)

        # Add penalty to the mu parameter output by the model to constrain it to the range(-pi, pi)
        penalty = 0.01 * K.mean(K.exp(K.square(self.model.output[0]) - K.square(np.pi)), axis=-1)

        updates = self.agent_optimizer.get_updates(self.model.trainable_weights, [], loss + K.mean(penalty))
        return K.function([self.model.input, self.action_pl, self.advantages_pl], [loss, eligibility, entropy, penalty], updates=updates)

    def sample(self, mu, scale):
        """ Draw a single sample from the model's MVN (for each entry in the batch) """
        normal_sample = np.random.normal(0.0, 1.0, size=mu.shape)
        sample = mu + np.einsum('bij,bj->bi', scale, normal_sample)

        return sample

    def sample_tf(self):
        """ Draw a single sample from the model's MVN (for each entry in the batch) """
        #mu_pl = K.placeholder(shape=(None, self.output_dim))
        #scale_pl = K.placeholder(shape=(None, self.output_dim, self.output_dim))
        mu_pl = self.model.output[0]
        scale_pl = self.model.output[1]

        normal_sample = K.random_normal(shape=mu_pl.shape, mean=0.0, stddev=1.0)
        print("normal_sample shape: " + str(normal_sample.shape))
        sample = mu_pl + K.dot(scale_pl, normal_sample)
        print("sample shape: " + str(sample.shape))

        return sample

    def entropy(self):
        """ Calculate the entropy (in bits) of the MVN """
        #mu_pl = K.placeholder(shape=(None, self.output_dim))
        #scale_pl = K.placeholder(shape=(None, self.output_dim, self.output_dim))
        mu_pl = self.model.output[0]
        scale_pl = self.model.output[1]

        # log(det(sigma)) = 2 * Sum_i(log(diag(L)_i))
        diag_part = K.tf.matrix_diag_part(scale_pl)
        print("diag_part shape: " + str(diag_part.shape))
        log_diag = K.log(diag_part)
        print("log_diag shape: " + str(log_diag.shape))
        log_sigma_det = 2 * K.sum(log_diag, axis=-1)
        print("log_sigma_det shape: " + str(log_sigma_det.shape))

        # Entropy = 0.5 * D * log(2pi*e) + 0.5 * log(det(sigma))
        pi = K.constant(np.pi)
        entropy = 0.5 * (self.output_dim *  K.log(2 * pi * K.exp(1.0)) + log_sigma_det)
        print("entropy shape: " + str(entropy.shape))
        #entropy = 0.5 * (K.tf.repeat((self.output_dim *  K.log(2 * pi * K.exp(1))), repeats=mu_pl.shape[0].value, axis=0) + log_sigma_det)

        return entropy

    def log_prob(self, action_pl):
        """ Return the log-probability of a given sample """
        #mu_pl = K.placeholder(shape=(None, self.output_dim))
        #scale_pl = K.placeholder(shape=(None, self.output_dim, self.output_dim))
        mu_pl = self.model.output[0]
        scale_pl = self.model.output[1]

        # Efficient calculation of (x - mu).T @ Sigma^-1 @ (x - mu)
        scale_inv = K.tf.matrix_inverse(scale_pl)
        print("scale_inv shape: " + str(scale_inv.shape))
        sample_minus_mu = action_pl - mu_pl
        print("sample_minus_mu shape: " + str(sample_minus_mu.shape))
        sample_minus_mu = K.expand_dims(sample_minus_mu, axis=-1)
        print("Expanded sample_minus_mu shape: " + str(sample_minus_mu.shape))
        intermediate_product = K.batch_dot(K.tf.matrix_transpose(scale_inv), sample_minus_mu)
        print("intermediate_product shape: " + str(intermediate_product.shape))
        sample_T_sigma_sample = K.squeeze(K.batch_dot(K.tf.matrix_transpose(intermediate_product), intermediate_product), axis=1)
        print("sample_T_sigma_sample shape: " + str(sample_T_sigma_sample.shape))

        # log(det(sigma)) = 2 * Sum_i(log(diag(L)_i))
        log_sigma_det = K.expand_dims(2 * K.sum(K.log(K.tf.matrix_diag_part(scale_pl)), axis=-1), axis=-1)
        print("log_sigma_det shape: " + str(log_sigma_det.shape))

        # log-likelihood of the sample = -0.5*(log(det(sigma)) + sample_T_sigma_sample + D * log(2pi))
        pi = K.constant(np.pi)
        log_prob = K.squeeze(-0.5*(log_sigma_det + sample_T_sigma_sample + self.output_dim * K.log(2*pi)), axis=-1)
        print("log_prob shape: " + str(log_prob.shape))
        #log_prob = -0.5*(log_sigma_det + sample_T_sigma_sample + K.tf.repeat(self.output_dim * K.log(2*pi), repeats=mu_pl.shape[0].value))

        return log_prob


    def save(self, path):
        self.model.save_weights(path + '_actor.hdf5')

    def load_weights(self, path):
        self.model.load_weights(path)


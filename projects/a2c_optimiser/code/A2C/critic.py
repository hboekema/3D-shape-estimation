import numpy as np
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.optimizers import Adam
from .agent import Agent

class Critic(Agent):
    """ Critic for the A2C Algorithm
    """

    def __init__(self, input_dim, output_dim, network, lr):
        Agent.__init__(self, input_dim, output_dim, lr, clipvalue=10)
        self.model = self.add_head(network)
        self.td_target = K.placeholder(shape=(None,))

        print(self.model.summary())

        # Count the number of trainable etc. weights
        #trainable_count = int(np.sum([K.count_params(w) for w in self.model.trainable_weights]))
        #non_trainable_count = int(np.sum([K.count_params(w) for w in self.model.non_trainable_weights]))

        #print('\nCritic')
        #print('--------------------------------------------------')
        #print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        #print('Trainable params: {:,}'.format(trainable_count))
        #print('Non-trainable params: {:,}'.format(non_trainable_count))
        #print('--------------------------------------------------')

    def add_head(self, network):
        """ Assemble critic network to predict value of each state """
        x = Dense(256, activation='relu', name="critic_head_slot")(network.output)
        #x = Dropout(0.5)(x)
        out = Dense(1, activation='linear', name="value_est")(x)
        return Model(network.input, out)

    def optimizer(self):
        """ Critic optimization: MSE/MAE over discounted rewards """
        critic_loss = K.mean(K.square(K.stop_gradient(self.td_target) - self.model.output))
        #critic_loss = K.mean(K.abs(K.stop_gradient(self.td_target) - self.model.output))
        updates = self.agent_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
        return K.function([self.model.input, self.td_target], [critic_loss], updates=updates)

    def save(self, path):
        self.model.save_weights(path + '_critic.hdf5')

    def load_weights(self, path):
        self.model.load_weights(path)

import numpy as np
from keras.optimizers import Adam

class Agent:
    """ Agent base class """
    def __init__(self, input_dim, output_dim, lr, clipvalue=10):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.agent_optimizer = Adam(lr=lr, clipvalue=clipvalue)

    def fit(self, agent_input, target):
        """ Perform one epoch of training to fit the appropriate gradients """
        self.model.fit(self.reshape(agent_input), target, epochs=1, verbose=0)

    def predict(self, agent_input):
        """ Critic value prediction """
        return self.model.predict(self.reshape(agent_input))

    def reshape(self, x):
        if len(x.shape) < len(self.input_dim) + 1:
            return np.expand_dims(x, axis=0)
        else:
            return x

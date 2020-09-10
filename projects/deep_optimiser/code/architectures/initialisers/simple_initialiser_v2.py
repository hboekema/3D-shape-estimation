
import keras.backend as K
import numpy as np

def emb_init_weights(emb_params):
    """ Embedding weights initialiser """
    #print("emb_params shape: " + str(emb_params.shape))
    def emb_init_wrapper(parameter):
        initial_weights = emb_params[:, :, parameter]
        #print("initial_weights shape: " + str(initial_weights.shape))
        def emb_init(shape):
            """ Initializer for the embedding layer """
            #print("shape: " + str(shape))
            init_weights = np.reshape(initial_weights, shape)
            #print("init_weights shape: " + str(init_weights.shape))
            init = K.constant(init_weights, dtype="float32")
            #print("init shape: " + str(init.shape))
            return init
        return emb_init
    return emb_init_wrapper



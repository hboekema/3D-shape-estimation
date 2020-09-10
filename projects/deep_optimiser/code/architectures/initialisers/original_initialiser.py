
import keras.backend as K
from keras.layers import Add, Multiply, Concatenate, Reshape
import numpy as np

def emb_init_weights(emb_params, period=None, distractor=np.pi, pose_offset={}, dist="uniform", reset_to_zero=False):
    """ Embedding weights initialiser """
    recognised_modes = ["uniform", "gaussian", "normal"]
    assert dist in recognised_modes

    def emb_init_wrapper(param, offset=False):
        def emb_init(shape):
            """ Initializer for the embedding layer """
            curr_emb_param = K.tf.gather(emb_params, param, axis=-1)
            curr_emb_param = K.tf.cast(curr_emb_param, dtype="float32")
            epsilon = 1e-5

            if reset_to_zero:
                #print(emb_params.shape)
                epsilon = K.constant(np.tile(epsilon, shape))
                #print(epsilon.shape)
                curr_emb_param = K.zeros_like(curr_emb_param)
                #print(curr_emb_param.shape)
                curr_emb_param = Add()([curr_emb_param, epsilon])
                #print(curr_emb_param.shape)
                #exit(1)
            elif offset or "param_{:02d}".format(param) in pose_offset.keys():
                if offset:
                    k = K.constant(distractor["param_{:02d}".format(param)])
                else:
                    k = K.constant(pose_offset["param_{:02d}".format(param)])

                if dist == "uniform":
                    offset_value = K.random_uniform(shape=[shape[0]], minval=-k, maxval=k, dtype="float32")
                elif dist == "normal" or dist == "gaussian":
                    offset_value = K.random_normal(shape=[shape[0]], mean=0.0, stddev=k, dtype="float32")

                #print(offset_value)
                #exit(1)
                if period is not None and shape[0] % period == 0:
                    block_size = shape[0] // period
                    #factors = Concatenate()([K.random_normal(shape=[block_size], mean=np.sqrt((i+1)/period), stddev=0.01) for i in range(period)])
                    factors = Concatenate()([K.random_normal(shape=[block_size], mean=np.sqrt(float(i+1)/period), stddev=0.01) for i in range(period)])
                    offset_value = Multiply()([offset_value, factors])

                curr_emb_param = Add()([curr_emb_param, offset_value])

            init = Reshape(target_shape=[shape[1]])(curr_emb_param)
            #print("init shape: " + str(init.shape))
            #exit(1)
            return init
        return emb_init
    return emb_init_wrapper




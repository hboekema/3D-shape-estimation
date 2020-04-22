
import numpy as np
import keras.backend as K
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, Lambda


def DenseValueNetwork(input_dim=(6890, 3)):
    """ Estimates the value function for the given state """
    # Current state
    state_input = Input(input_dim, name="valnet_input")
    # Keep salient points only
    # In order of: right hand, right wrist, right forearm, right bicep end, right bicep, right shoulder, top of cranium, left shoulder, left bicep, left bicep end, left forearm, left wrist, left hand,
    # chest, belly/belly button, back of neck, upper back, central back, lower back/tailbone,
    # left foot, left over-ankle, left shin, left over-knee, left quadricep, left hip, right, hip, right, quadricep, right over-knee, right shin, right, over-ankle, right foot
    vertex_list = [5674, 5705, 5039, 5151, 4977, 4198, 411, 606, 1506, 1682, 1571, 2244, 2212,
            3074, 3500, 460, 2878, 3014, 3021,
            3365, 4606, 4588, 4671, 6877, 1799, 5262, 3479, 1187, 1102, 1120, 6740]
    #face_array = np.array([11396, 8620, 7866, 5431, 6460, 1732, 4507])
    vertex_diff = Lambda(lambda x: K.tf.gather(x, np.array(vertex_list).astype(np.int32), axis=-2))(state_input)
    print("vertex_diff shape: " + str(vertex_diff.shape))
    vertex_diff = Flatten()(vertex_diff)

    # Simple dense network with two hidden layers estimates the value function
    #valnet_architecture = Dense(2048, activation="relu")(vertex_diff)
    valnet_architecture = Dense(1024, activation="relu")(vertex_diff)
    #valnet_architecture = Dropout(0.5)(valnet_architecture)

    # Input is the state s, output is a base for the (estimated) expectation of return R
    return state_input, valnet_architecture


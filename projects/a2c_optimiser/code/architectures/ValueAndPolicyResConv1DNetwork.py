
import keras.backend as K
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, Lambda, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape


def ValueAndPolicyResConv1DNetwork(input_dim=(6890, 3)):
    """ Estimates the (stochastic) policy and value functions for the given state """
    # Current state - this is the difference between two point clouds
    state_input = Input(input_dim, name="polnet_input")

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

    # Dense network for estimating the value function
    valnet_architecture = Dense(2048, activation="relu")(vertex_diff)
    valnet_architecture = Dropout(0.5)(valnet_architecture)
    valnet_architecture = Dense(512, activation="relu")(valnet_architecture)
    valnet_architecture = Dropout(0.5)(valnet_architecture)
    valnet_output = Dense(1, activation="linear", name="return_estimate")(valnet_architecture)

    # 1D convolutional network for estimating the policy function
    # Input block
    polnet_input = Dense(128, activation="relu")(vertex_diff)
    polnet_input = BatchNormalization()(polnet_input)
    polnet_input = Reshape(polnet_input.shape[1], 1)(polnet_input)
    print('polnet_input shape: '+str(polnet_input.shape))
    polnet_input = Conv1D(64, 3, activation="relu", padding="same")(polnet_input)
    polnet_input = BatchNormalization()(polnet_input)
    print('polnet_input shape: '+str(polnet_input.shape))

    # Residual block 1
    polnet_architecture = Conv1D(64, 3, activation="relu", padding="same")(polnet_input)
    polnet_architecture = BatchNormalization()(polnet_architecture)
    polnet_architecture = Conv1D(64, 3, activation="linear", padding="same")(polnet_architecture)
    polnet_architecture = BatchNormalization()(polnet_architecture)
    #polnet_architecture = Conv1D(64, 1, activation="linear", padding="same", use_bias=False)(polnet_architecture)
    polnet_input = Conv1D(64, 1, activation="linear", padding="same", use_bias=False)(polnet_input)
    res1_output = Add()([polnet_architecture, polnet_input])
    res1_output = Activation("relu")(res1_output)
    print("res1_output shape: " + str(res1_output.shape))
    res1_output = MaxPooling1D(2)(res1_output)
    print("res1_output shape (after pooling): " + str(res1_output.shape))

    # Residual block 2
    polnet_architecture = Conv1D(128, 3, activation="relu", padding="same")(res1_output)
    polnet_architecture = BatchNormalization()(polnet_architecture)
    polnet_architecture = Conv1D(128, 3, activation="linear", padding="same")(polnet_architecture)
    polnet_architecture = BatchNormalization()(polnet_architecture)
    #polnet_architecture = Conv1D(128, 1, activation="linear", padding="same", use_bias=False)(polnet_architecture)
    res1_output = Conv1D(128, 1, activation="linear", padding="same", use_bias=False)(res1_output)
    res2_output = Add()([polnet_architecture, res1_output])
    res2_output = Activation("relu")(res2_output)
    print("res2_output shape: " + str(res2_output.shape))
    res2_output = MaxPooling1D(2)(res2_output)
    print("res2_output shape (after pooling): " + str(res2_output.shape))

    # Residual block 3
    polnet_architecture = Conv1D(256, 3, activation="relu", padding="same")(res2_output)
    polnet_architecture = BatchNormalization()(polnet_architecture)
    polnet_architecture = Conv1D(256, 3, activation="linear", padding="same")(polnet_architecture)
    polnet_architecture = BatchNormalization()(polnet_architecture)
    polnet_architecture = Conv1D(256, 1, activation="linear", padding="same", use_bias=False)(polnet_architecture)
    res2_output = Conv1D(256, 1, activation="linear", padding="same", use_bias=False)(res2_output)
    res3_output = Add()([polnet_architecture, res2_output])
    res3_output = Activation("relu")(res3_output)
    print("res3_output shape: " + str(res3_output.shape))
    #res3_output = MaxPooling1D(2)(res3_output)
    #print("res3_output shape (after pooling): " + str(res3_output.shape))

    # Output block
    polnet_architecture = GlobalAveragePooling1D()(res3_output)
    print("polnet architecture shape (after GAP): " + str(polnet_architecture.shape))
    polnet_architecture = Dense(128, activation="relu")(polnet_architecture)
    polnet_architecture = BatchNormalization()(polnet_architecture)
    print('polnet_architecture shape: '+str(polnet_architecture.shape))
    polnet_mu = Dense(85, activation="linear", name="polnet_mu")(polnet_architecture)
    polnet_sigma = Dense(85, activation="softplus", name="polnet_sigma")(polnet_architecture)   # can (in theory) be zero - need to guard by adding small constant
    policy_dist = Lambda(lambda x: K.tf.contrib.distributions.Normal(x[0], x[1]), name="policy_dist")([polnet_mu, polnet_sigma])

    # Draw a sample from the policy distribution for this state
    policy_sample = Lambda(lambda x: x.sample(1), name="policy_sample")(policy_dist)
    policy_sample = Reshape((85,), name="policy_sample_output")(policy_sample)

    # Input is the state s, outputs are the sample from the policy distribution, the distribution itself and the estimated expectation of the return
    return [state_input], [policy_sample, policy_dist, valnet_output]


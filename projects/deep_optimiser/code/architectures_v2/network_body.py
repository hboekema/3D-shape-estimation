

""" Standard networks """

def Conv1D4L(deep_opt_input, DROPOUT=0.0):
    print("Network body: Conv1D4L")
    optlearner_architecture = Conv1D(64, 5, strides=2, activation="relu")(deep_opt_input)
    optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
    optlearner_architecture = Conv1D(128, 5, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
    optlearner_architecture = Conv1D(256, 3, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
    optlearner_architecture = Conv1D(512, 3, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    optlearner_architecture = Reshape((-1,))(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))

    delta_d_hat = Dense(85, activation="linear", name="delta_d_hat")(optlearner_architecture)
    print('delta_d_hat shape: '+str(delta_d_hat.shape))

    return delta_d_hat



""" Grouped network """

def grouped_network(deep_opt_input, groups, DROPOUT=0.0):
    deep_opt_input = Reshape((deep_opt_input.shape[1].value, 1))(deep_opt_input)
    print('deep_opt_input shape: '+str(deep_opt_input.shape))
    indices_ordering = []
    group_outputs = []
    for group in groups:
        optlearner_architecture = Conv1D(64, 5, strides=2, activation="relu")(deep_opt_input)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv1D(128, 5, strides=2, activation="relu")(optlearner_architecture)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv1D(256, 3, strides=2, activation="relu")(optlearner_architecture)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv1D(512, 3, strides=2, activation="relu")(optlearner_architecture)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        optlearner_architecture = Reshape((-1,))(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))

        delta_d_hat = Dense(3*len(group), activation="linear")(optlearner_architecture)
        print('delta_d_hat shape: ' + str(delta_d_hat.shape))
        group_outputs.append(delta_d_hat)

        indices = []
        for joint in group:
            j_base = 3*joint
            j1 = j_base
            j2 = j_base + 1
            j3 = j_base + 2
            indices += [j1, j2, j3]
        indices_ordering += indices

    return indices_ordering, group_outputs


def grouped_6D_network(deep_opt_input, groups, DROPOUT=0.0):
    deep_opt_input = Reshape((deep_opt_input.shape[1].value, 1))(deep_opt_input)
    print('deep_opt_input shape: '+str(deep_opt_input.shape))
    indices_ordering = []
    group_outputs = []
    for group in groups:
        optlearner_architecture = Conv1D(64, 5, strides=2, activation="relu")(deep_opt_input)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv1D(128, 5, strides=2, activation="relu")(optlearner_architecture)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv1D(256, 3, strides=2, activation="relu")(optlearner_architecture)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv1D(512, 3, strides=2, activation="relu")(optlearner_architecture)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        optlearner_architecture = Reshape((-1,))(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))

        delta_d_hat = Dense(6*len(group), activation="linear")(optlearner_architecture)
        print('delta_d_hat shape: ' + str(delta_d_hat.shape))
        group_outputs.append(delta_d_hat)

        indices = []
        for joint in group:
            j_base = 3*joint
            j1 = j_base
            j2 = j_base + 1
            j3 = j_base + 2
            indices += [j1, j2, j3]
        indices_ordering += indices

    return indices_ordering, group_outputs


def shape_network(deep_opt_input, DROPOUT=0.0):
    optlearner_architecture = Conv1D(64, 5, strides=2, activation="relu")(deep_opt_input)
    optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
    optlearner_architecture = Conv1D(128, 5, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
    optlearner_architecture = Conv1D(256, 3, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
    optlearner_architecture = Conv1D(512, 3, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    optlearner_architecture = Reshape((-1,))(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    shape_params = Dense(13, activation="linear", name="shape_params")(optlearner_architecture)
    print('shape_params shape: '+str(shape_params.shape))

    return shape_params





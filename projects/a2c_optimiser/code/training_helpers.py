
import numpy as np

def format_distractor_dict(k, params):
    """ Format the distractor values to the accepted format """
    if isinstance(k, (int, float, str)):
        k_temp = k
        k = {"other": k_temp}
    if isinstance(k, dict):
        keys = k.keys()
        if "other" not in keys:
            k["other"] = 0.0
        for key, value in k.iteritems():
            if value == "pi":
                k[key] = np.pi
        for param in params:
            if param not in keys:
                k[param] = k["other"]

    del k["other"]
    return k


def offset_params(X_params, params_to_offset, DISTRACTOR=np.pi):
    """ Apply initial offset k to params_to_offset in X_params """
    if isinstance(DISTRACTOR, (int, float)):
        k = {param: DISTRACTOR for param in params_to_offset}
    else:
        # k must be a dict with an entry for each variable parameter
        k = DISTRACTOR

    offset_params_int = [int(param[6:8]) for param in params_to_offset]
    data_samples = X_params.shape[0]
    X_params[:, offset_params_int] = np.array([k[param] * (1 - 2*np.random.rand(data_samples)) for param in params_to_offset]).T

    return X_params


def update_weights_wrapper(DISTRACTOR, num_samples, PERIOD, trainable_params, optlearner_model, gt_data=None, generator=None):
    """ Wrapper for weight update functionality """
    # Set the distractor value
    if isinstance(DISTRACTOR, (int, float)):
        k = {param: DISTRACTOR for param in trainable_params}
    else:
        # k must be a dict with an entry for each variable parameter
        k = DISTRACTOR

    def update_weights(epoch, logs):
        # May need to verify that this works correctly - visually it looks okay but something might be wrong
        # Update a block of parameters
        BL_SIZE = num_samples // PERIOD
        #print("epoch: " + str(epoch))
        BL_INDEX = epoch % PERIOD
        #print("BL_SIZE: " + str(BL_SIZE))
        #print("BL_INDEX: " + str(BL_INDEX))

        if generator is not None:
            gt_data, _ = generator.yield_data()
            gt_params = gt_data[1]
            #print("Gt params shape: " + str(gt_params.shape))
            #print("\n------------------\n" + str(gt_params[0]) + "\n------------------\n")
            param_ids = ["param_{:02d}".format(i) for i in range(85)]
            not_trainable = [param for param in param_ids if param not in trainable_params]
            for param in not_trainable:
                layer = optlearner_model.get_layer(param)
                weights = gt_params[:, int(param[6:8])].reshape((1, num_samples, 1))
                layer.set_weights(weights)

        for param in trainable_params:
            layer = optlearner_model.get_layer(param)
            weights = np.array(layer.get_weights())
            #print("weights " + str(weights))
            #print('weights shape: '+str(weights.shape))
            #exit(1)
            #weights_new = [ (1-2*np.random.rand(weights[i].shape[0], weights[i].shape[1])) * k for i in range(len(weights))]
            #weights_new = [ (1-2*np.random.rand(BL_SIZE, weights[i].shape[1])) * k for i in range(weights.shape[0])]
            weights_new = [ (1-2*np.random.rand(BL_SIZE, weights[i].shape[1])) * k[param] for i in range(weights.shape[0])]
            #print("weights new " + str(weights_new))
            weights[:, BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE] = weights_new    # only update the required block of weights
            layer.set_weights(weights)
            #exit(1)



    def update_weights_artificially(epoch, logs):
        for param in trainable_params:
            gt_weights = gt_data[:, int(param[6:8])]
            layer = optlearner_model.get_layer(param)
            weights = np.array(layer.get_weights())
            #print("weights " + str(weights))
            #print('weights shape: '+str(weights.shape))
            #exit(1)

            # Compute the offset from the ground truth value
            offset_value = np.random.uniform(low=-k[param], high=k[param], size=(num_samples, 1))
            #offset_value = K.random_uniform(shape=[num_samples], minval=-k[param], maxval=k[param], dtype="float32")
            block_size = num_samples // PERIOD
            means = [np.sqrt(float((i + 1 + epoch) % PERIOD)/PERIOD) for i in range(PERIOD)]
            np.random.shuffle(means)
            factors = np.concatenate([np.random.normal(loc=means[i], scale=0.01, size=(block_size, 1)) for i in range(PERIOD)])
            offset_value *= factors
            #print("offset_value shape: " + str(offset_value.shape))
            new_weights = gt_weights.reshape(offset_value.shape) + offset_value
            new_weights = new_weights.reshape(weights.shape)
            #print("new_weights shape: " + str(new_weights.shape))
            #factors = Concatenate()([K.random_normal(shape=[block_size], mean=means[i], stddev=0.01) for i in range(PERIOD)])
            #offset_value = Multiply()([offset_value, factors])
            #weights = Add()([gt_weights, offset_value])

            layer.set_weights(new_weights)
            #exit(1)

    if gt_data is None:
        return update_weights
    else:
        return update_weights_artificially



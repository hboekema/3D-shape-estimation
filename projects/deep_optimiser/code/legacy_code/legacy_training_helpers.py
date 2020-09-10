
import numpy as np
from architectures.OptLearnerCombinedStaticModArchitecture import OptLearnerCombinedStaticModArchitecture
from architectures.OptLearnerMeshNormalStaticArchitecture import OptLearnerMeshNormalStaticArchitecture
from architectures.OptLearnerMeshNormalStaticModArchitecture import OptLearnerMeshNormalStaticModArchitecture
from architectures.BasicFCOptLearnerStaticArchitecture import BasicFCOptLearnerStaticArchitecture
from architectures.FullOptLearnerStaticArchitecture import FullOptLearnerStaticArchitecture
from architectures.Conv1DFullOptLearnerStaticArchitecture import Conv1DFullOptLearnerStaticArchitecture
from architectures.GAPConv1DOptLearnerStaticArchitecture import GAPConv1DOptLearnerStaticArchitecture
from architectures.DeepConv1DOptLearnerStaticArchitecture import DeepConv1DOptLearnerStaticArchitecture
from architectures.NewDeepConv1DOptLearnerArchitecture import NewDeepConv1DOptLearnerArchitecture
from architectures.ResConv1DOptLearnerStaticArchitecture import ResConv1DOptLearnerStaticArchitecture
from architectures.ProbCNNOptLearnerStaticArchitecture import ProbCNNOptLearnerStaticArchitecture
from architectures.GatedCNNOptLearnerArchitecture import GatedCNNOptLearnerArchitecture
from architectures.LatentConv1DOptLearnerStaticArchitecture import LatentConv1DOptLearnerStaticArchitecture
from architectures.RotConv1DOptLearnerArchitecture import RotConv1DOptLearnerArchitecture
from architectures.ConditionalOptLearnerArchitecture import ConditionalOptLearnerArchitecture
from architectures.GroupedConv1DOptLearnerArchitecture import GroupedConv1DOptLearnerArchitecture


def format_distractor_dict(k, trainable_params):
    """ Format the distractor values to the accepted format """
    all_params = ["param_{:02d}".format(value) for value in range(85)]
    pose_params = ["param_{:02d}".format(value) for value in range(72)]
    shape_params = ["param_{:02d}".format(value) for value in range(72, 82)]
    trans_params = ["param_{:02d}".format(value) for value in range(82, 85)]
    if isinstance(k, (int, float, str)):
        k_temp = k
        k = {"other": k_temp}
    if isinstance(k, dict):
        keys = k.keys()
        if "other" not in keys:
            k["other"] = 0.0
        if "trainable" not in keys:
            k["trainable"] = k["other"]
        if "pose_other" not in keys:
            k["pose_other"] = 0.0
        if "shape_other" not in keys:
            k["shape_other"] = 0.0
        if "trans_other" not in keys:
            k["trans_other"] = 0.0

        for key, value in k.iteritems():
            if value == "pi":
                k[key] = np.pi
        for param in all_params:
            if param not in keys:
                if param in pose_params:
                    if param in trainable_params:
                        k[param] = k["trainable"]
                    else:
                        k[param] = k["pose_other"]
                elif param in shape_params:
                    k[param] = k["shape_other"]
                elif param in trans_params:
                    k[param] = k["trans_other"]
                else:
                    k[param] = k["other"]

    del k["trainable"]
    del k["pose_other"]
    del k["shape_other"]
    del k["trans_other"]
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


def update_weights_wrapper(DISTRACTOR, num_samples, PERIOD, trainable_params, optlearner_model, gt_data=None, generator=None, offset_nt=False, pose_offset={}):
    """ Wrapper for weight update functionality """
    # Set the distractor value
    if isinstance(DISTRACTOR, (int, float)):
        k = {param: DISTRACTOR for param in trainable_params}
    else:
        # k must be a dict with an entry for each variable parameter
        k = DISTRACTOR
    #print(k)
    #print(pose_offset)
    #exit(1)

    def update_weights(epoch, logs):
        # Update a block of parameters
        BL_SIZE = num_samples // PERIOD
        #print("epoch: " + str(epoch))
        BL_INDEX = epoch % PERIOD
        #print("BL_SIZE: " + str(BL_SIZE))
        #print("BL_INDEX: " + str(BL_INDEX))

        if generator is not None:
            gt_data, _ = generator.yield_data(epoch)
            gt_params = gt_data[1]
            #print("Gt params shape: " + str(gt_params.shape))
            #print("\n------------------\n" + str(gt_params[0]) + "\n------------------\n")
            param_ids = ["param_{:02d}".format(i) for i in range(85)]
            not_trainable = [param for param in param_ids if param not in trainable_params]
            for param in not_trainable:
                layer = optlearner_model.get_layer(param)
                weights = np.array(layer.get_weights())
                weights[:, BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE] = gt_params[BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE, int(param[6:8])].reshape((1, BL_SIZE, 1))
                if offset_nt and param in pose_offset.keys():
                    extra_offset = [ (1-2*np.random.rand(BL_SIZE, weights[i].shape[1])) * pose_offset[param] for i in range(weights.shape[0])]
                    weights[:, BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE] += extra_offset
                layer.set_weights(weights)

#        param_ids = ["param_{:02d}".format(i) for i in range(85)]
#        not_trainable = [param for param in param_ids if param not in trainable_params]
#        for param in not_trainable:
#            layer = optlearner_model.get_layer(param)
#            weights = np.array(layer.get_weights())
#            if offset_nt and param in pose_offset.keys():
#                offset = [ (1-2*np.random.rand(BL_SIZE, weights[i].shape[1])) * pose_offset[param] for i in range(weights.shape[0])]
#                weights[:, BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE] = offset
#            layer.set_weights(weights)

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


def architecture_inputs_and_outputs(ARCHITECTURE, param_trainable, emb_initialiser, smpl_params, input_info, faces, data_samples, INPUT_TYPE):
    if ARCHITECTURE == "OptLearnerMeshNormalStaticModArchitecture":
        optlearner_inputs, optlearner_outputs = OptLearnerMeshNormalStaticModArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "BasicFCOptLearnerStaticArchitecture":
        optlearner_inputs, optlearner_outputs = BasicFCOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "FullOptLearnerStaticArchitecture":
        optlearner_inputs, optlearner_outputs = FullOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "Conv1DFullOptLearnerStaticArchitecture":
        optlearner_inputs, optlearner_outputs = Conv1DFullOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "GAPConv1DOptLearnerStaticArchitecture":
        optlearner_inputs, optlearner_outputs = GAPConv1DOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "DeepConv1DOptLearnerStaticArchitecture":
        optlearner_inputs, optlearner_outputs = DeepConv1DOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture":
        optlearner_inputs, optlearner_outputs = NewDeepConv1DOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "ResConv1DOptLearnerStaticArchitecture":
        optlearner_inputs, optlearner_outputs = ResConv1DOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "ProbCNNOptLearnerStaticArchitecture":
        optlearner_inputs, optlearner_outputs = ProbCNNOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "GatedCNNOptLearnerArchitecture":
        optlearner_inputs, optlearner_outputs = GatedCNNOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "LatentConv1DOptLearnerStaticArchitecture":
        optlearner_inputs, optlearner_outputs = LatentConv1DOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "RotConv1DOptLearnerArchitecture":
        optlearner_inputs, optlearner_outputs = RotConv1DOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "ConditionalOptLearnerArchitecture":
        optlearner_inputs, optlearner_outputs = ConditionalOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    elif ARCHITECTURE == "GroupedConv1DOptLearnerArchitecture":
        optlearner_inputs, optlearner_outputs = GroupedConv1DOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
    else:
        raise ValueError("Architecture '{}' not recognised".format(ARCHITECTURE))

    return optlearner_inputs, optlearner_outputs



def architecture_output_array(ARCHITECTURE, data_samples, num_trainable=24):
    """ Return correct output for each architecture """
    Y_data = []

    if ARCHITECTURE == "OptLearnerMeshNormalStaticModArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 7))]
    elif ARCHITECTURE == "BasicFCOptLearnerStaticArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 7))]
    elif ARCHITECTURE == "FullOptLearnerStaticArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31))]
    elif ARCHITECTURE == "Conv1DFullOptLearnerStaticArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31))]
    elif ARCHITECTURE == "GAPConv1DOptLearnerStaticArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31))]
    elif ARCHITECTURE == "DeepConv1DOptLearnerStaticArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31))]
    elif ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture":
        #Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31))]
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31)), np.zeros((data_samples, 85))]
        #Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 37)), np.zeros((data_samples, 85))]
    elif ARCHITECTURE == "ResConv1DOptLearnerStaticArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31))]
    elif ARCHITECTURE == "ProbCNNOptLearnerStaticArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples,))]
    elif ARCHITECTURE == "GatedCNNOptLearnerArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples,))]
    elif ARCHITECTURE == "LatentConv1DOptLearnerStaticArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31))]
    elif ARCHITECTURE == "RotConv1DOptLearnerArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31)), np.zeros((data_samples, 1, 3, 3)), np.zeros((data_samples, 24, 3, 3)), np.zeros((data_samples, 24*3*2)), np.zeros((data_samples, 24*3*2))]
        #Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31)), np.zeros((data_samples, 1, 3, 3)), np.zeros((data_samples, 24, 3, 3)), np.zeros((data_samples, 24*3*2)), np.zeros((data_samples, 24*3*2)), np.zeros((data_samples, 24, 3)), np.zeros((data_samples, 24, 3))]
    elif ARCHITECTURE == "ConditionalOptLearnerArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31)), np.zeros((data_samples, 1, 3, 3)), np.zeros((data_samples, num_trainable, 3, 3)), np.zeros((data_samples, num_trainable*3*2)), np.zeros((data_samples, num_trainable*3*2))]
    elif ARCHITECTURE == "GroupedConv1DOptLearnerArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31)), np.zeros((data_samples, 85))]
    else:
        raise ValueError("Architecture '{}' not recognised".format(ARCHITECTURE))

    return Y_data


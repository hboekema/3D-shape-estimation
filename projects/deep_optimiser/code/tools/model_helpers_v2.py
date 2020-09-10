
import os
import sys
import numpy as np

from keras.models import Model

#from architectures.OptLearnerCombinedStaticModArchitecture import OptLearnerCombinedStaticModArchitecture
#from architectures.OptLearnerMeshNormalStaticArchitecture import OptLearnerMeshNormalStaticArchitecture
#from architectures.OptLearnerMeshNormalStaticModArchitecture import OptLearnerMeshNormalStaticModArchitecture
#from architectures.BasicFCOptLearnerStaticArchitecture import BasicFCOptLearnerStaticArchitecture
#from architectures.FullOptLearnerStaticArchitecture import FullOptLearnerStaticArchitecture
#from architectures.Conv1DFullOptLearnerStaticArchitecture import Conv1DFullOptLearnerStaticArchitecture
#from architectures.GAPConv1DOptLearnerStaticArchitecture import GAPConv1DOptLearnerStaticArchitecture
#from architectures.DeepConv1DOptLearnerStaticArchitecture import DeepConv1DOptLearnerStaticArchitecture
##from architectures.NewDeepConv1DOptLearnerArchitecture import NewDeepConv1DOptLearnerArchitecture
#from architectures.NewDeepConv1DOptLearnerArchitecture_v2 import NewDeepConv1DOptLearnerArchitecture
#from architectures.ResConv1DOptLearnerStaticArchitecture import ResConv1DOptLearnerStaticArchitecture
#from architectures.ProbCNNOptLearnerStaticArchitecture import ProbCNNOptLearnerStaticArchitecture
#from architectures.GatedCNNOptLearnerArchitecture import GatedCNNOptLearnerArchitecture
#from architectures.LatentConv1DOptLearnerStaticArchitecture import LatentConv1DOptLearnerStaticArchitecture
#from architectures.RotConv1DOptLearnerArchitecture import RotConv1DOptLearnerArchitecture
#from architectures.ConditionalOptLearnerArchitecture import ConditionalOptLearnerArchitecture
from architectures.GroupedConv1DOptLearnerArchitecture_v3 import GroupedConv1DOptLearnerArchitecture
from architectures.GroupedConv2DOptLearnerArchitecture import GroupedConv2DOptLearnerArchitecture
from architectures.ShapeConv2DOptLearnerArchitecture import ShapeConv2DOptLearnerArchitecture
#from architectures.PeriodicOptLearnerArchitecture import PeriodicOptLearnerArchitecture
from architectures.architecture_helpers import no_loss, false_loss, false_loss_summed, load_smpl_params


def architecture_inputs_and_outputs(ARCHITECTURE, param_trainable, emb_initialiser, smpl_params, input_info, faces, data_samples, INPUT_TYPE, groups=[], update_weight=1.0, DROPOUT=0.0, PARAM_2D_TRAINABLE=False, include_shape=True):
#    if ARCHITECTURE == "OptLearnerMeshNormalStaticModArchitecture":
#        optlearner_inputs, optlearner_outputs = OptLearnerMeshNormalStaticModArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
#    elif ARCHITECTURE == "BasicFCOptLearnerStaticArchitecture":
#        optlearner_inputs, optlearner_outputs = BasicFCOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
#    elif ARCHITECTURE == "FullOptLearnerStaticArchitecture":
#        optlearner_inputs, optlearner_outputs = FullOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
#    elif ARCHITECTURE == "Conv1DFullOptLearnerStaticArchitecture":
#        optlearner_inputs, optlearner_outputs = Conv1DFullOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
#    elif ARCHITECTURE == "GAPConv1DOptLearnerStaticArchitecture":
#        optlearner_inputs, optlearner_outputs = GAPConv1DOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
#    elif ARCHITECTURE == "DeepConv1DOptLearnerStaticArchitecture":
#        optlearner_inputs, optlearner_outputs = DeepConv1DOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
#    elif ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture":
#        optlearner_inputs, optlearner_outputs = NewDeepConv1DOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
#    elif ARCHITECTURE == "ResConv1DOptLearnerStaticArchitecture":
#        optlearner_inputs, optlearner_outputs = ResConv1DOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
#    elif ARCHITECTURE == "ProbCNNOptLearnerStaticArchitecture":
#        optlearner_inputs, optlearner_outputs = ProbCNNOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
#    elif ARCHITECTURE == "GatedCNNOptLearnerArchitecture":
#        optlearner_inputs, optlearner_outputs = GatedCNNOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
#    elif ARCHITECTURE == "LatentConv1DOptLearnerStaticArchitecture":
#        optlearner_inputs, optlearner_outputs = LatentConv1DOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
#    elif ARCHITECTURE == "RotConv1DOptLearnerArchitecture":
#        optlearner_inputs, optlearner_outputs = RotConv1DOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
#    elif ARCHITECTURE == "ConditionalOptLearnerArchitecture":
#        optlearner_inputs, optlearner_outputs = ConditionalOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE, groups=groups)
    #elif ARCHITECTURE == "GroupedConv1DOptLearnerArchitecture":
    if ARCHITECTURE == "GroupedConv1DOptLearnerArchitecture":
        optlearner_inputs, optlearner_outputs = GroupedConv1DOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE, groups=groups)
    elif ARCHITECTURE == "GroupedConv2DOptLearnerArchitecture":
        optlearner_inputs, optlearner_outputs = GroupedConv2DOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, PARAM_2D_TRAINABLE=PARAM_2D_TRAINABLE, emb_size=data_samples, input_type=INPUT_TYPE, groups=groups, DROPOUT=DROPOUT)
    elif ARCHITECTURE == "ShapeConv2DOptLearnerArchitecture":
        optlearner_inputs, optlearner_outputs = ShapeConv2DOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, PARAM_2D_TRAINABLE=PARAM_2D_TRAINABLE, emb_size=data_samples, input_type=INPUT_TYPE, groups=groups, DROPOUT=DROPOUT, include_shape=include_shape)
#    elif ARCHITECTURE == "PeriodicOptLearnerArchitecture":
#        optlearner_inputs, optlearner_outputs = PeriodicOptLearnerArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE, groups=groups, update_weight=update_weight)
    else:
        raise ValueError("Architecture '{}' not recognised".format(ARCHITECTURE))

    return optlearner_inputs, optlearner_outputs


def construct_optlearner_model(ARCHITECTURE, param_trainable, emb_initialiser, data_samples, INPUT_TYPE, GROUPS=[], update_weight=1.0, DROPOUT=0.0, PARAM_2D_TRAINABLE=False, INCLUDE_SHAPE=True):
    smpl_params, input_info, faces = load_smpl_params()
    print("Optimiser architecture: " + str(ARCHITECTURE))

    optlearner_inputs, optlearner_outputs = architecture_inputs_and_outputs(ARCHITECTURE, param_trainable, emb_initialiser, smpl_params, input_info, faces, data_samples, INPUT_TYPE, GROUPS, update_weight, DROPOUT, PARAM_2D_TRAINABLE, INCLUDE_SHAPE)
    print("optlearner inputs " +str(optlearner_inputs))
    print("optlearner outputs "+str(optlearner_outputs))
    optlearner_model = Model(inputs=optlearner_inputs, outputs=optlearner_outputs)

    return optlearner_model


def get_trainable_layers(optlearner_model, param_trainable):
    trainable_layers_names = [param_layer for param_layer, trainable in param_trainable.items() if trainable]
    trainable_layers = {optlearner_model.get_layer(layer_name): int(layer_name[6:8]) for layer_name in trainable_layers_names}

    return trainable_layers


def freeze_layers(optlearner_model, param_trainable):
    trainable_layers_names = [param_layer for param_layer, trainable in param_trainable.items() if trainable]
    for layer in optlearner_model.layers:
        if layer.name not in trainable_layers_names:
            layer.trainable = False

    return optlearner_model


def initialise_emb_layers(optlearner_model, param_trainable, initial_weights):
    for layer_name, trainable in param_trainable.items():
        param_number = int(layer_name[6:8])
        emb_layer = optlearner_model.get_layer(layer_name)
        emb_layer.set_weights(initial_weights[:, param_number].reshape((1, -1, 1)))
        #print(np.array(emb_layer.get_weights()).shape)

    return optlearner_model


def gather_optlearner_summed_losses(INPUT_TYPE, ARCHITECTURE, LOSS_WEIGHTS=[1.0, 1.0, 1.0, 0.0, 0.0], BATCH_SIZE=1):
    if INPUT_TYPE == "MESH_NORMALS" or INPUT_TYPE == "ONLY_NORMALS":
        optlearner_loss = [no_loss,
                false_loss_summed, # delta_d loss (L_smpl loss)
                no_loss,
                no_loss, # point cloud loss (L_xent)
                false_loss_summed, # delta_d hat loss (L_delta_smpl)
                no_loss, # delta_d_hat sin metric
                false_loss_summed, # this is the loss which updates smpl parameter inputs with predicted gradient
                no_loss,
                no_loss,
                false_loss_summed # difference angle loss (L_xent)
                ]
        #print(str(LOSS_WEIGHTS))
        #exit(1)
        optlearner_loss_weights=[
                0.0,
                LOSS_WEIGHTS[0]*BATCH_SIZE, # delta_d loss (L_smpl loss)
                0.0,
                0.0, # point cloud loss (L_xent)
                LOSS_WEIGHTS[2]*BATCH_SIZE, # delta_d_hat loss (L_delta_smpl)
                0.0, # delta_d_hat sin metric - always set to 0
                LOSS_WEIGHTS[3], # this is the loss which updates smpl parameter inputs with predicted gradient
                0.0,
                0.0,
                LOSS_WEIGHTS[1]*BATCH_SIZE, # difference angle loss (L_xent)
                ]

    elif INPUT_TYPE == "3D_POINTS":
        #assert False, "gt_normals loss case is not currently handled"
        optlearner_loss = [no_loss,
                false_loss_summed, # delta_d loss (L_smpl loss)
                no_loss,
                false_loss_summed, # point cloud loss (L_xent)
                false_loss_summed, # delta_d hat loss (L_delta_smpl)
                no_loss, # delta_d_hat sin metric
                false_loss_summed, # this is the loss which updates smpl parameter inputs with predicted gradient
                no_loss, no_loss,
                no_loss # difference angle loss (L_xent)
                ]
        optlearner_loss_weights=[
                0.0,
                LOSS_WEIGHTS[0]*BATCH_SIZE, # delta_d loss (L_smpl loss)
                0.0,
                LOSS_WEIGHTS[1]*BATCH_SIZE, # point cloud loss (L_xent)
                LOSS_WEIGHTS[2]*BATCH_SIZE, # delta_d_hat loss (L_delta_smpl)
                0.0, # delta_d_hat sin metric - always set to 0
                LOSS_WEIGHTS[3], # this is the loss which updates smpl parameter inputs with predicted gradient
                0.0, 0.0,
                0.0, # difference angle loss (L_xent)
                ]
    else:
        assert False, "input type not recognised"

    if ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture" or ARCHITECTURE == "ConditionalOptLearnerArchitecture" or ARCHITECTURE == "PeriodicOptLearnerArchitecture":
        optlearner_loss += [false_loss]
        optlearner_loss_weights += [0.0]

    if ARCHITECTURE == "ProbCNNOptLearnerStaticArchitecture" or ARCHITECTURE == "GatedCNNOptLearnerArchitecture" or ARCHITECTURE == "GroupedConv1DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss]
        optlearner_loss_weights += [0.0, 0.0, 0.0]

    if ARCHITECTURE == "RotConv1DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss, false_loss]
        optlearner_loss_weights += [0.0, 0.0, 0.0, 0.0]

    if  ARCHITECTURE == "GroupedConv2DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss, false_loss_summed]
        optlearner_loss_weights += [0.0,
                0.0,
                0.0,
                LOSS_WEIGHTS[4]*BATCH_SIZE     # gt_normals_LOSS loss for Conv2D network
                ]

    if  ARCHITECTURE == "ShapeConv2DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss, false_loss_summed, no_loss, no_loss, no_loss]
        optlearner_loss_weights += [0.0,
                0.0,
                0.0,
                LOSS_WEIGHTS[4]*BATCH_SIZE,     # gt_normals_LOSS loss for Conv2D network
                0.0,
                0.0,
                0.0
                ]

    return optlearner_loss, optlearner_loss_weights


def gather_optlearner_losses(INPUT_TYPE, ARCHITECTURE, LOSS_WEIGHTS=[1.0, 1.0, 1.0, 0.0, 0.0], BATCH_SIZE=1):
    if INPUT_TYPE == "MESH_NORMALS" or INPUT_TYPE == "ONLY_NORMALS":
        optlearner_loss = [no_loss,
                false_loss, # delta_d loss (L_smpl loss)
                no_loss,
                no_loss, # point cloud loss (L_xent)
                false_loss, # delta_d hat loss (L_delta_smpl)
                no_loss, # delta_d_hat sin metric
                false_loss_summed, # this is the loss which updates smpl parameter inputs with predicted gradient
                no_loss,
                no_loss,
                false_loss # difference angle loss (L_xent)
                ]
        #print(str(LOSS_WEIGHTS))
        #exit(1)
        optlearner_loss_weights=[
                0.0,
                LOSS_WEIGHTS[0]*BATCH_SIZE, # delta_d loss (L_smpl loss)
                0.0,
                0.0, # point cloud loss (L_xent)
                LOSS_WEIGHTS[2]*BATCH_SIZE, # delta_d_hat loss (L_delta_smpl)
                0.0, # delta_d_hat sin metric - always set to 0
                LOSS_WEIGHTS[3], # this is the loss which updates smpl parameter inputs with predicted gradient
                0.0,
                0.0,
                LOSS_WEIGHTS[1]*BATCH_SIZE, # difference angle loss (L_xent)
                ]

    elif INPUT_TYPE == "3D_POINTS":
        #assert False, "gt_normals loss case is not currently handled"
        optlearner_loss = [no_loss,
                false_loss, # delta_d loss (L_smpl loss)
                no_loss,
                false_loss, # point cloud loss (L_xent)
                false_loss, # delta_d hat loss (L_delta_smpl)
                no_loss, # delta_d_hat sin metric
                false_loss_summed, # this is the loss which updates smpl parameter inputs with predicted gradient
                no_loss, no_loss,
                no_loss # difference angle loss (L_xent)
                ]
        optlearner_loss_weights=[
                0.0,
                LOSS_WEIGHTS[0]*BATCH_SIZE, # delta_d loss (L_smpl loss)
                0.0,
                LOSS_WEIGHTS[1]*BATCH_SIZE, # point cloud loss (L_xent)
                LOSS_WEIGHTS[2]*BATCH_SIZE, # delta_d_hat loss (L_delta_smpl)
                0.0, # delta_d_hat sin metric - always set to 0
                LOSS_WEIGHTS[3], # this is the loss which updates smpl parameter inputs with predicted gradient
                0.0, 0.0,
                0.0, # difference angle loss (L_xent)
                ]
    else:
        assert False, "input type not recognised"

    if ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture" or ARCHITECTURE == "ConditionalOptLearnerArchitecture" or ARCHITECTURE == "PeriodicOptLearnerArchitecture":
        optlearner_loss += [false_loss]
        optlearner_loss_weights += [0.0]

    if ARCHITECTURE == "ProbCNNOptLearnerStaticArchitecture" or ARCHITECTURE == "GatedCNNOptLearnerArchitecture" or ARCHITECTURE == "GroupedConv1DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss]
        optlearner_loss_weights += [0.0, 0.0, 0.0]

    if ARCHITECTURE == "RotConv1DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss, false_loss]
        optlearner_loss_weights += [0.0, 0.0, 0.0, 0.0]

    if  ARCHITECTURE == "GroupedConv2DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss, false_loss]
        optlearner_loss_weights += [0.0,
                0.0,
                0.0,
                LOSS_WEIGHTS[4]*BATCH_SIZE     # gt_normals_LOSS loss for Conv2D network
                ]

    if  ARCHITECTURE == "ShapeConv2DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss, false_loss_summed, no_loss, no_loss, no_loss]
        optlearner_loss_weights += [0.0,
                0.0,
                0.0,
                LOSS_WEIGHTS[4]*BATCH_SIZE,     # gt_normals_LOSS loss for Conv2D network
                0.0,
                0.0,
                0.0
                ]

    return optlearner_loss, optlearner_loss_weights


def gather_test_optlearner_losses(ARCHITECTURE, LOSS_WEIGHTS=[1.0, 1.0, 1.0]):
    optlearner_loss = [no_loss,
                no_loss, # delta_d loss (L_smpl loss)
                no_loss,
                false_loss_summed, # point cloud loss (L_xent)
                no_loss, # delta_d hat loss (L_delta_smpl)
                no_loss, # delta_d_hat sin metric
                false_loss_summed, # this is the loss which updates smpl parameter inputs with predicted gradient
                no_loss, no_loss,
                false_loss_summed # difference angle loss (L_xent)
                ]
    optlearner_loss_weights=[
                0.0,
                0.0, # delta_d loss (L_smpl loss)
                0.0,
                LOSS_WEIGHTS[0], # point cloud loss (L_xent)
                0.0, # delta_d_hat loss (L_delta_smpl)
                0.0, # delta_d_hat sin metric - always set to 0
                LOSS_WEIGHTS[2], # this is the loss which updates smpl parameter inputs with predicted gradient
                0.0, 0.0,
                LOSS_WEIGHTS[1], # difference angle loss (L_xent)
                ]

    if ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture" or ARCHITECTURE == "ConditionalOptLearnerArchitecture" or ARCHITECTURE == "PeriodicOptLearnerArchitecture":
        optlearner_loss += [false_loss]
        optlearner_loss_weights += [0.0]

    if ARCHITECTURE == "ProbCNNOptLearnerStaticArchitecture" or ARCHITECTURE == "GatedCNNOptLearnerArchitecture" or ARCHITECTURE == "GroupedConv1DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss]
        optlearner_loss_weights += [0.0, 0.0, 0.0]

    if ARCHITECTURE == "RotConv1DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss, false_loss]
        optlearner_loss_weights += [0.0, 0.0, 0.0, 0.0]

    if  ARCHITECTURE == "GroupedConv2DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss, false_loss]
        optlearner_loss_weights += [0.0,
                0.0,
                0.0,
                0.0     # gt_normals_LOSS loss for Conv2D network
                ]

    if  ARCHITECTURE == "ShapeConv2DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss, false_loss, no_loss, no_loss, no_loss]
        optlearner_loss_weights += [0.0,
                0.0,
                0.0,
                0.0,     # gt_normals_LOSS loss for Conv2D network
                0.0,
                0.0,
                0.0
                ]
    return optlearner_loss, optlearner_loss_weights


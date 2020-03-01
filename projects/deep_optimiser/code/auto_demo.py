import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import cv2
from keras.models import Model
from keras.optimizers import Adam, SGD
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import copy
import yaml

from callbacks import OptLearnerPredOnEpochEnd
from silhouette_generator import OptLearnerDataGenerator, OptLearnerExtraOutputDataGenerator
from architectures.OptLearnerCombinedStaticModArchitecture import OptLearnerCombinedStaticModArchitecture
from architectures.OptLearnerMeshNormalStaticArchitecture import OptLearnerMeshNormalStaticArchitecture
from architectures.OptLearnerMeshNormalStaticModArchitecture import OptLearnerMeshNormalStaticModArchitecture
from architectures.BasicFCOptLearnerStaticArchitecture import BasicFCOptLearnerStaticArchitecture
from architectures.FullOptLearnerStaticArchitecture import FullOptLearnerStaticArchitecture
from architectures.Conv1DFullOptLearnerStaticArchitecture import Conv1DFullOptLearnerStaticArchitecture
from architectures.GAPConv1DOptLearnerStaticArchitecture import GAPConv1DOptLearnerStaticArchitecture
from architectures.DeepConv1DOptLearnerStaticArchitecture import DeepConv1DOptLearnerStaticArchitecture
from architectures.ResConv1DOptLearnerStaticArchitecture import ResConv1DOptLearnerStaticArchitecture
from architectures.LatentConv1DOptLearnerStaticArchitecture import LatentConv1DOptLearnerStaticArchitecture
from architectures.architecture_helpers import false_loss, no_loss, load_smpl_params, emb_init_weights
from generate_data import load_data
from smpl_np import SMPLModel
from render_mesh import Mesh
from euler_rodrigues_transform import rodrigues_to_euler
from training_helpers import offset_params, format_distractor_dict

parser = ArgumentParser()
#parser.add_argument("--model", help="Path to the model weights")
parser.add_argument("--config", help="Path to the data directory")
parser.add_argument("--data", help="Path to the data directory")
args = parser.parse_args()

# Read in the configurations
if args.config is not None:
    with open(args.config, 'r') as f:
        setup_params = json.load(f)
else:
    with open("./config.yaml", 'r') as f:
        try:
            setup_params = yaml.safe_load(f)
            #print(setup_params)
        except yaml.YAMLError as exc:
            print(exc)


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = setup_params["GENERAL"]["GPU_ID"]
print("GPU used: |" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")


""" Test set-up """

#np.random.seed(10)
np.random.seed(11)

exp_dir = os.getcwd().replace("code", "")
print("Experiment directory: " + str(exp_dir))
models = os.listdir(exp_dir + "models/")
if len(models) == 0:
    print("No models for this experiment. Exiting.")
    exit(1)
else:
    models.sort(key=lambda x: float(x[x.find("-")+1 : x.find(".hdf5")]))
    model_name = models[0]
    print("Using model '{}'".format(model_name))

save_suffix = ""
#save_suffix = "_non-zero_pose"
model = exp_dir + "models/" + model_name
logs_dir = exp_dir + "logs/" + model_name + save_suffix + "/"
os.system('mkdir ' + logs_dir)
control_logs_dir = exp_dir + "logs/control" + save_suffix + "/"
os.system('mkdir ' + control_logs_dir)
test_vis_dir = exp_dir + "test_vis/" + model_name + save_suffix + "/"
os.system('mkdir ' + test_vis_dir)
control_dir = exp_dir + "test_vis/" + "control" + save_suffix + "/"
os.system('mkdir ' + control_dir)


# Generate the data from the SMPL parameters
trainable_params = setup_params["PARAMS"]["TRAINABLE"]
param_ids = ["param_{:02d}".format(i) for i in range(85)]

if trainable_params == "all_pose":
    not_trainable = [0, 1, 2]
    #not_trainable = []
    #trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 72]
    trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 66]
    trainable_params = [param_ids[index] for index in trainable_params_indices]
elif trainable_params == "all_pose_and_global_rotation":
    not_trainable = [0, 2]
    #not_trainable = []
    #trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 72]
    trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 66]
    trainable_params = [param_ids[index] for index in trainable_params_indices]

#trainable_params = ["param_14", "param_17", "param_59", "param_56"]
#trainable_params = ["param_01", "param_59", "param_56"]
#trainable_params = ["param_01", "param_59"]
#trainable_params = ["param_59", "param_56"]
#trainable_params = ["param_01"]
#trainable_params = ["param_56"]
#trainable_params = ["param_59"]

param_trainable = { param: (param in trainable_params) for param in param_ids }


# Basic experimental setup
RESET_PERIOD = setup_params["BASIC"]["RESET_PERIOD"]
MODEL_SAVE_PERIOD = setup_params["BASIC"]["MODEL_SAVE_PERIOD"]
PREDICTION_PERIOD = setup_params["BASIC"]["PREDICTION_PERIOD"]
OPIMISATION_PERIOD = setup_params["BASIC"]["OPTIMISATION_PERIOD"]
MODE = setup_params["BASIC"]["ROT_MODE"]
DISTRACTOR = setup_params["BASIC"]["DISTRACTOR"]
data_samples = setup_params["BASIC"]["NUM_SAMPLES"]
LOAD_DATA_DIR = setup_params["DATA"]["TEST_DATA_DIR"]
POSE_OFFSET = setup_params["DATA"]["POSE_OFFSET"]
PARAMS_TO_OFFSET = setup_params["DATA"]["PARAMS_TO_OFFSET"]
ARCHITECTURE = setup_params["MODEL"]["ARCHITECTURE"]
INPUT_TYPE = setup_params["MODEL"]["INPUT_TYPE"]
TEST_EPOCHS = setup_params["TEST"]["test_iterations"]

# Format the distractor and offset dictionaries
DISTRACTOR = format_distractor_dict(DISTRACTOR, trainable_params)
if PARAMS_TO_OFFSET == "trainable_params":
    PARAMS_TO_OFFSET = trainable_params
elif PARAMS_TO_OFFSET == "all_pose_and_global_rotation":
    not_trainable = [0, 2]
    #not_trainable = []
    #trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 72]
    trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 66]
    PARAMS_TO_OFFSET = [param_ids[index] for index in trainable_params_indices]
POSE_OFFSET = format_distractor_dict(POSE_OFFSET, PARAMS_TO_OFFSET)

# Generate the data from the SMPL parameters
smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
zero_params = np.zeros(shape=(85,))
zero_pc = smpl.set_params(beta=zero_params[72:82], pose=zero_params[0:72].reshape((24,3)), trans=zero_params[82:85])
#print("zero_pc: " + str(zero_pc))

# Generate and format the data
X_indices = np.array([i for i in range(data_samples)])
X_params = np.array([zero_params for i in range(data_samples)], dtype="float32")
if LOAD_DATA_DIR is not None:
    X_params, X_pcs = load_data(LOAD_DATA_DIR, num_samples=data_samples, load_silhouettes=False)
else:
    if not all(value == 0.0 for value in POSE_OFFSET.values()):
        X_params = offset_params(X_params, PARAMS_TO_OFFSET, POSE_OFFSET)
        X_pcs = np.array([smpl.set_params(beta=params[72:82], pose=params[0:72], trans=params[82:85]) for params in X_params])
    else:
        X_pcs = np.array([zero_pc for i in range(data_samples)], dtype="float32")

if MODE == "EULER":
    # Convert from Rodrigues to Euler angles
    X_params = rodrigues_to_euler(X_params, smpl)

X_data = [np.array(X_indices), np.array(X_params), np.array(X_pcs)]
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
elif ARCHITECTURE == "ResConv1DOptLearnerStaticArchitecture":
    Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31))]
elif ARCHITECTURE == "LatentConv1DOptLearnerStaticArchitecture":
    Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31))]
else:
    raise ValueError("Architecture '{}' not recognised".format(ARCHITECTURE))

x_test = X_data
y_test = Y_data

# Render silhouettes for the callback data
num_samples = 5
cb_indices = X_indices[:num_samples]
cb_params = X_params[:num_samples]
cb_pcs = X_pcs[:num_samples]
X_cb = [np.array(cb_indices), np.array(cb_params), np.array(cb_pcs)]
silh_cb = []
for pc in cb_pcs:
    silh = Mesh(pointcloud=pc).render_silhouette(show=False)
    silh_cb.append(silh)

# Initialise the embedding layer
def emb_init_weights_np(emb_params, distractor=np.pi):
    def emb_init_wrapper(param, offset=False):
        def emb_init(shape, dtype="float32"):
            """ Initializer for the embedding layer """
            emb_params_ = emb_params[:, param]

            if offset:
                k = distractor
                offset_ = k["param_{:02d}".format(param)] * 2 * (np.random.rand(shape[0]) - 0.5)
                emb_params_[:] += offset_

            init = np.array(emb_params_, dtype=dtype).reshape(shape)
            #print("init shape: " + str(init.shape))
            #print("init values: " + str(init))
            #exit(1)
            return init
        return emb_init
    return emb_init_wrapper

emb_initialiser = emb_init_weights_np(X_params, distractor=DISTRACTOR)

# Load model
smpl_params, input_info, faces = load_smpl_params()
print("Optimiser architecture: " + str(ARCHITECTURE))
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
elif ARCHITECTURE == "ResConv1DOptLearnerStaticArchitecture":
    optlearner_inputs, optlearner_outputs = ResConv1DOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
elif ARCHITECTURE == "LatentConv1DOptLearnerStaticArchitecture":
    optlearner_inputs, optlearner_outputs = LatentConv1DOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples, input_type=INPUT_TYPE)
else:
    raise ValueError("Architecture '{}' not recognised".format(ARCHITECTURE))
#input_indices = [0, 2]
#output_indices = [0, 2, 3, 5]
#optlearner_model = Model(inputs=[input_ for i, input_ in enumerate(optlearner_inputs) if i in input_indices], outputs=[output for i, output in enumerate(optlearner_outputs) if i in output_indices])
optlearner_model = Model(inputs=optlearner_inputs, outputs=optlearner_outputs)
optlearner_model.load_weights(model)

# Freeze all layers except for the required embedding layers
trainable_layers_names = [param_layer for param_layer, trainable in param_trainable.items() if trainable]
trainable_layers = {optlearner_model.get_layer(layer_name): int(layer_name[6:8]) for layer_name in trainable_layers_names}
for layer in optlearner_model.layers:
    if layer.name not in trainable_layers_names:
        layer.trainable = False


# Set the weights of the embedding layer
for layer_name, trainable in param_trainable.items():
    emb_layer = optlearner_model.get_layer(layer_name)
    emb_init_ = emb_initialiser(param=int(layer_name[6:8]), offset=trainable)
    emb_layer.set_weights(emb_init_(shape=(data_samples, 1)).reshape(1, data_samples, 1))
    #print(np.array(emb_layer.get_weights()).shape)

# Compile the model
learning_rate = 0.01
optimizer = Adam(lr=learning_rate, decay=0.0)
#optimizer = SGD(learning_rate, momentum=0.0, nesterov=False)
#optlearner_model.compile(optimizer=optimizer, loss=[no_loss, false_loss, no_loss, false_loss,
#                                                    false_loss,  # pc loss
#                                                    false_loss,  # loss that updates smpl parameters
#                                                    no_loss, no_loss, no_loss],
#                                            loss_weights=[0.0, 0.0, 0.0,
#                                                        1.0*1.0, # point cloud loss
#                                                        0.0*10, # delta_d_hat loss
#                                                        0.0, # this is the loss which updates smpl parameter inputs with predicted gradient
#                                                        0.0, 0.0, 0.0])
optlearner_model.compile(optimizer=optimizer, loss=[no_loss,
                                                   false_loss, # delta_d loss (L_smpl loss)
                                                   no_loss,
                                                   false_loss, # point cloud loss (L_xent)
                                                   no_loss, # delta_d hat loss (L_delta_smpl)
                                                   no_loss, # delta_d_hat sin metric
						   no_loss, # this is the loss which updates smpl parameter inputs with predicted gradient
						   no_loss, no_loss,
                                                   no_loss # difference angle loss (L_xent)
                                                   #no_loss, no_loss, no_loss
                                                   ],
                                            loss_weights=[0.0,
                                                        0.0*1, # delta_d loss (L_smpl loss)
                                                        0.0,
                                                        0.0*1, # point cloud loss (L_xent)
                                                        0.0*1, # delta_d_hat loss (L_delta_smpl)
                                                        0.0, # delta_d_hat sin metric - always set to 0
                                                        0.0, # this is the loss which updates smpl parameter inputs with predicted gradient
                                                        0.0, 0.0,
                                                        0.0*1, # difference angle loss (L_xent)
                                                        #0.0, 0.0, 0.0
                                                        ],
                                            metrics={"delta_d_hat_sin_output": false_loss}
                                            #options=run_options,
                                            #run_metadata=run_metadata
                                            )

# Print model summary
optlearner_model.summary()


# Visualisation callback
epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=test_vis_dir, period=1, trainable_params=trainable_params, visualise=False, testing=True)
epoch_pred_cb_control = OptLearnerPredOnEpochEnd(control_logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=control_dir, period=1, trainable_params=trainable_params,visualise=False, testing=True)


def learned_optimizer(optlearner_model, epochs=50, lr=0.1, mode="RODRIGUES"):
    metrics_names = optlearner_model.metrics_names
    print("metrics_names: " + str(metrics_names))
    named_scores = {}
    output_names = [output.op.name.split("/")[0] for output in optlearner_model.outputs]
    print("output_names: " + str(output_names))
    pred_index = output_names.index("delta_d_hat")
    #print("prediction index: " + str(pred_index))

    epoch_pred_cb.set_model(optlearner_model)
    #layer_names = ["diff_cross_product", "diff_angle_mse"]
    #intermediate_layer_model = Model(inputs=optlearner_model.input, outputs=[optlearner_model.get_layer(layer_name).output for layer_name in layer_names])

    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1))
        print("----------------------")
        epoch_pred_cb.on_epoch_begin(epoch=int(epoch), logs=named_scores)
        #print('emb layer weights'+str(emb_weights))
	#print('shape '+str(emb_weights[0].shape))
	y_pred = optlearner_model.predict(x_test)
        delta_d_hat = y_pred[pred_index]

        # Evaluate the mesh normals
        #intermediate_output = intermediate_layer_model.predict(x_test)
        #print("mesh_normals: " + str(intermediate_output[0][:5]))
        #print("mesh_normals_mse: " + str(intermediate_output[1][:5]))

        for emb_layer, param_num in trainable_layers.items():
	    emb_weights = emb_layer.get_weights()
            #emb_weights += lr * np.array(y_pred[7][:, param_num]).reshape((data_samples, 1))
            #emb_weights += lr * np.array(y_pred[8][:, param_num]).reshape((data_samples, 1))
            emb_weights += lr * np.array(delta_d_hat[:, param_num]).reshape((data_samples, 1))
	    emb_layer.set_weights(emb_weights)

        # Evaluate model performance
        #scores = optlearner_model.evaluate(x_test, y_test, batch_size=100)
        #for i, score in enumerate(scores):
        #    named_scores[metrics_names[i]] = score
        #print("scores: " + str(scores))
        #exit(1)
        epoch_pred_cb.on_epoch_end(epoch=int(epoch), logs=named_scores)
    epoch_pred_cb.set_model(optlearner_model)


def regular_optimizer(optlearner_model, epochs=50):
    # Train the model
    optlearner_model.fit(
                x_test,
                y_test,
                batch_size=32,
                epochs=epochs,
                callbacks=[epoch_pred_cb_control]
            )


if __name__ == "__main__":
    learned_optimizer(optlearner_model, epochs=TEST_EPOCHS, lr=0.5)
    #regular_optimizer(optlearner_model, epochs=TEST_EPOCHS)
    exit(1)


# Evaluate the model performance
eval_log = optlearner_model.evaluate_generator(test_gen)

eval_string = ""
for i in range(len(eval_log)):
    eval_string += str(optlearner_model.metrics_names[i]) + ": " + str(eval_log[i]) + "  "
print(eval_string)

# Predict the parameters from the silhouette and generate a mesh
X_test, Y_test = test_gen.__getitem__(0)
prediction = optlearner_model.predict(X_test)
for pred in prediction:
    print(pred.shape)

pred_params = prediction[0]
real_params = Y_test[0]

for pred in pred_params:
    pointcloud = smpl.set_params(pred[10:82], pred[0:10], pred[82:85])
    pred_silhouette = Mesh(pointcloud=pointcloud).render_silhouette()

    pointcloud = smpl.set_params(real_params[10:82], real_params[0:10], real_params[82:85])
    real_silhouette = Mesh(pointcloud=pointcloud).render_silhouette()

    diff_silhouette = abs(real_silhouette - pred_silhouette)

    comp_silh = np.concatenate([real_silhouette, pred_silhouette, diff_silhouette])
    plt.imshow(comp_silh, cmap='gray')
    plt.show()




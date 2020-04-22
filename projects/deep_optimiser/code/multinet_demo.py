import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import cv2
import yaml
from datetime import datetime

# Parse the command-line arguments
parser = ArgumentParser()
parser.add_argument("--run_id", help="Identifier of this network pass")

args = parser.parse_args()

exp_dirs = [
        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-21_15:30:19/",
        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-21_15:37:33/",
        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-21_15:38:18/",
        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-21_15:38:53/",
        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-21_15:40:00/",
        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-21_15:40:38/"
        ]

# Read in the configurations
all_setup_params = []
for exp_dir in exp_dirs:
    with open(exp_dir + "code/config.yaml", 'r') as f:
        try:
            all_setup_params.append(yaml.safe_load(f))
            #print(setup_params)
        except yaml.YAMLError as exc:
            print(exc)

# Set the ID of this training pass
if args.run_id is not None:
    run_id = args.run_id
else:
    # Use the current date, time and model architecture as a run-id
    run_id = datetime.now().strftime("{}_%Y-%m-%d_%H:%M:%S".format("MultiNet"))
    run_id = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/{}/".format(run_id)
os.system('mkdir ' + str(run_id))

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

setup_params = all_setup_params[0]

# Set number of GPUs to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = setup_params["GENERAL"]["GPU_ID"]
print("gpu used:|" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")
#exit(1)

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

from keras.models import Model
from keras.optimizers import Adam, SGD
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import pickle
import copy

from callbacks import OptLearnerPredOnEpochEnd
from silhouette_generator import OptLearnerDataGenerator, OptLearnerExtraOutputDataGenerator
from architectures.architecture_helpers import false_loss, no_loss, load_smpl_params, emb_init_weights
from generate_data import load_data
from smpl_np import SMPLModel
from render_mesh import Mesh
from euler_rodrigues_transform import rodrigues_to_euler
from training_helpers import offset_params, format_distractor_dict, architecture_output_array, architecture_inputs_and_outputs



""" Test set-up """

#np.random.seed(10)
np.random.seed(11)

models = []
for exp_dir in exp_dirs:
    print("Experiment directory: " + str(exp_dir))
    models_in_dir = os.listdir(exp_dir + "models/")
    if len(models_in_dir) == 0:
        print("No models for this experiment. Exiting.")
        exit(1)
    else:
        models_in_dir.sort(key=lambda x: float(x[x.find("-")+1 : x.find(".hdf5")]))
        model_name = models_in_dir[0]
        print("Using model '{}'".format(model_name))

    model = exp_dir + "models/" + model_name
    models.append(model)

learning_rates = [0.500]

model_dir = run_id + "models/"
logs_dir = run_id + "logs/"
opt_logs_dir = logs_dir + "opt/"
train_vis_dir = run_id + "train_vis/"
test_vis_dir = run_id + "test_vis/"
opt_vis_dir = run_id + "opt_vis/"
code_dir = run_id + "code/"
tensorboard_logs_dir = logs_dir + "scalars/"
os.mkdir(model_dir)
os.mkdir(logs_dir)
os.mkdir(opt_logs_dir)
os.mkdir(train_vis_dir)
os.mkdir(test_vis_dir)
os.mkdir(opt_vis_dir)
os.mkdir(code_dir)
print("Experiment directory: \n" + str(exp_dir))
os.system("cp -r ./* " + str(code_dir))

save_suffix = ""
#save_suffix = "_non-zero_pose"
logs_dir = run_id + "logs/model" + save_suffix + "/"
os.system('mkdir ' + logs_dir)
control_logs_dir = run_id + "logs/control" + save_suffix + "/"
os.system('mkdir ' + control_logs_dir)
test_vis_dir = run_id + "test_vis/model" + save_suffix + "/"
os.system('mkdir ' + test_vis_dir)
control_dir = run_id + "test_vis/" + "control" + save_suffix + "/"
os.system('mkdir ' + control_dir)

logs_dirs = []
control_logs_dirs = []
test_vis_dirs = []
control_dirs = []
for lr in learning_rates:
    #save_suffix = "_lr_" + str(lr)
    sub_dir = "lr_{:02f}/".format(lr)
    logs_dir = run_id + "logs/model" + save_suffix + "/" + sub_dir
    logs_dirs.append(logs_dir)
    os.system('mkdir ' + logs_dir)
    control_logs_dir = run_id + "logs/control" + save_suffix + "/" + sub_dir
    control_logs_dirs.append(control_logs_dir)
    os.system('mkdir ' + control_logs_dir)
    test_vis_dir = run_id + "test_vis/model" + save_suffix + "/" + sub_dir
    test_vis_dirs.append(test_vis_dir)
    os.system('mkdir ' + test_vis_dir)
    control_dir = run_id + "test_vis/" + "control" + save_suffix + "/" + sub_dir
    control_dirs.append(control_dir)
    os.system('mkdir ' + control_dir)


# Generate the data from the SMPL parameters
#trainable_params = setup_params["PARAMS"]["TRAINABLE"]
trainable_params = "all_3d"
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
elif trainable_params == "all_3d":
    trainable_params = param_ids[:72]

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
print("loading SMPL")
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
        num_test_samples = 5
        print("Offsetting parameters")
        X_params = offset_params(X_params, PARAMS_TO_OFFSET, POSE_OFFSET)
        X_params = X_params[:num_test_samples]
        print("X_params shape: " + str(X_params.shape))
        print("Rendering parameters")
        X_pcs = np.array([np.array(smpl.set_params(beta=params[72:82], pose=params[0:72].reshape((24, 3)), trans=params[82:85]).copy()) for params in X_params])
        print("X_pcs shape: " + str(X_pcs.shape))
        assert data_samples % num_test_samples == 0
        X_pcs = X_pcs.copy()
        X_pcs = np.array([X_pcs for i in range(int(data_samples/num_test_samples))], dtype=np.float32)
        print("X_pcs shape: " + str(X_pcs.shape))
        X_pcs = np.reshape(X_pcs, (data_samples, 6890, 3))
        #X_pcs = np.repeat(X_pcs, data_samples/num_test_samples)
        print("X_pcs shape: " + str(X_pcs.shape))
        X_params = np.array([X_params for i in range(data_samples/num_test_samples)], dtype=np.float32).reshape((data_samples, 85))
        print("X_params shape: " + str(X_params.shape))
    else:
        X_pcs = np.array([zero_pc for i in range(data_samples)], dtype="float32")

if MODE == "EULER":
    # Convert from Rodrigues to Euler angles
    X_params = rodrigues_to_euler(X_params, smpl)

X_data = [np.array(X_indices), np.array(X_params), np.array(X_pcs)]
Y_data = architecture_output_array(ARCHITECTURE, data_samples)

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

# Set the weights of the embedding layer
initial_weights = np.zeros((data_samples, 85))
for layer_name, trainable in param_trainable.items():
    emb_init_ = emb_initialiser(param=int(layer_name[6:8]), offset=trainable)
    layer_init_weights = emb_init_(shape=(data_samples,))
    initial_weights[:, int(layer_name[6:8])] = layer_init_weights

# Build models
optlearner_models = []
optlearner_layers = []
for i, model in enumerate(models):
    setup_params = all_setup_params[i]
    trainable_params = setup_params["PARAMS"]["TRAINABLE"]

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
    elif trainable_params == "all_3d":
        trainable_params = param_ids[:72]

    param_trainable = { param: (param in trainable_params) for param in param_ids }

    # Load models
    smpl_params, input_info, faces = load_smpl_params()
    print("Optimiser architecture: " + str(ARCHITECTURE))
    optlearner_inputs, optlearner_outputs = architecture_inputs_and_outputs(ARCHITECTURE, param_trainable, emb_initialiser, smpl_params, input_info, faces, data_samples, INPUT_TYPE)
    print("optlearner inputs " +str(optlearner_inputs))
    print("optlearner outputs "+str(optlearner_outputs))

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
    optlearner_layers.append(trainable_layers_names)

    # Set the weights of the embedding layer
    for layer_name, trainable in param_trainable.items():
        emb_layer = optlearner_model.get_layer(layer_name)
        emb_layer.set_weights(layer_init_weights.reshape((1, data_samples, 1)))
        #print(np.array(emb_layer.get_weights()).shape)

    # Compile the model
    learning_rate = 0.01
    optimizer = Adam(lr=learning_rate, decay=0.0)
    #optimizer = SGD(learning_rate, momentum=0.0, nesterov=False)
    if INPUT_TYPE == "MESH_NORMALS":
        optlearner_loss = [no_loss,
                false_loss, # delta_d loss (L_smpl loss)
                no_loss,
                no_loss, # point cloud loss (L_xent)
                false_loss, # delta_d hat loss (L_delta_smpl)
                no_loss, # delta_d_hat sin metric
                no_loss, # this is the loss which updates smpl parameter inputs with predicted gradient
                no_loss, no_loss,
                false_loss # difference angle loss (L_xent)
                ]
        optlearner_loss_weights=[
                0.0,
                1.0, # delta_d loss (L_smpl loss)
                0.0,
                0.0, # point cloud loss (L_xent)
                1.0, # delta_d_hat loss (L_delta_smpl)
                0.0, # delta_d_hat sin metric - always set to 0
                0.0/learning_rate, # this is the loss which updates smpl parameter inputs with predicted gradient
                0.0, 0.0,
                1.0, # difference angle loss (L_xent)
                ]

    elif INPUT_TYPE == "3D_POINTS":
        optlearner_loss = [no_loss,
                false_loss, # delta_d loss (L_smpl loss)
                no_loss,
                false_loss, # point cloud loss (L_xent)
                false_loss, # delta_d hat loss (L_delta_smpl)
                no_loss, # delta_d_hat sin metric
                no_loss, # this is the loss which updates smpl parameter inputs with predicted gradient
                no_loss, no_loss,
                no_loss # difference angle loss (L_xent)
                ]
        optlearner_loss_weights=[
                0.0,
                1.0, # delta_d loss (L_smpl loss)
                0.0,
                1.0, # point cloud loss (L_xent)
                1.0, # delta_d_hat loss (L_delta_smpl)
                0.0, # delta_d_hat sin metric - always set to 0
                0.0/learning_rate, # this is the loss which updates smpl parameter inputs with predicted gradient
                0.0, 0.0,
                0.0, # difference angle loss (L_xent)
                ]

    if ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture":
        optlearner_loss += [false_loss]
        optlearner_loss_weights += [0.0]

    if ARCHITECTURE == "RotConv1DOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss, false_loss]
        optlearner_loss_weights += [0.0, 0.0, 0.0, 0.0]
        #optlearner_loss += [false_loss, false_loss, false_loss, false_loss, false_loss, false_loss]
        #optlearner_loss_weights += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if ARCHITECTURE == "ProbCNNOptLearnerStaticArchitecture" or ARCHITECTURE == "GatedCNNOptLearnerArchitecture":
        optlearner_loss += [false_loss, false_loss, false_loss]
        optlearner_loss_weights += [0.0, 0.0, 0.0]

    optlearner_model.compile(optimizer=optimizer, loss=optlearner_loss, loss_weights=optlearner_loss_weights,
                                                metrics={"delta_d_hat_sin_output": false_loss}
                                                #metrics={"delta_d_hat_sin_output": trainable_param_metric([int(param[6:8]) for param in trainable_params])}
                                                #options=run_options,
                                                #run_metadata=run_metadata
                                                )

    # Print model summary
    optlearner_model.summary()
    optlearner_models.append(optlearner_model)

# Visualisation callbacks
epoch_pred_cbs = []
epoch_pred_cbs_control = []
for lr_num, lr in enumerate(learning_rates):
    logs_dir = logs_dirs[lr_num]
    test_vis_dir = test_vis_dirs[lr_num]
    control_logs_dir = control_logs_dirs[lr_num]
    control_dir = control_dirs[lr_num]
    epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=test_vis_dir, period=1, trainable_params=trainable_params, visualise=False, testing=True)
    epoch_pred_cbs.append(epoch_pred_cb)
    epoch_pred_cb_control = OptLearnerPredOnEpochEnd(control_logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=control_dir, period=1, trainable_params=trainable_params,visualise=False, testing=True)
    epoch_pred_cbs_control.append(epoch_pred_cb_control)


def learned_optimizer(optlearner_models, epochs=50, lr=0.1, cb=epoch_pred_cb, mode="RODRIGUES"):
    layers_names = [param_layer for param_layer, trainable in param_trainable.items() if int(param_layer[6:8] < 72)]
    optlearner_models_layers = [{optlearner_model.get_layer(layer_name): int(layer_name[6:8]) for layer_name in layers_names} for optlearner_model in optlearner_models]

    metrics_names = optlearner_models[0].metrics_names
    print("metrics_names: " + str(metrics_names))
    named_scores = {}
    output_names = [output.op.name.split("/")[0] for output in optlearner_models[0].outputs]
    print("output_names: " + str(output_names))
    pred_index = output_names.index("delta_d_hat")
    #print("prediction index: " + str(pred_index))

    cb.set_model(optlearner_models[0])

    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1))
        print("----------------------")
        cb.on_epoch_begin(epoch=int(epoch), logs=named_scores)
        #print('emb layer weights'+str(emb_weights))
	#print('shape '+str(emb_weights[0].shape))
        test_samples = [arr[:num_samples] for arr in x_test]
        for i, optlearner_model in enumerate(optlearner_models):
            y_pred = optlearner_model.predict(test_samples)
            delta_d_hat = np.zeros((data_samples, 85))
            delta_d_hat[:num_samples] = y_pred[pred_index]

            # Apply this update to all models' weights
            for optlearner_model_layer in optlearner_models_layers:
                for emb_layer, param_num in optlearner_model_layer:
                    if "param_{:02d}".param_num in optlearner_layers[i]:
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
        cb.on_epoch_end(epoch=int(epoch), logs=named_scores)
    cb.set_model(optlearner_models[0])


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
    for lr_num, lr in enumerate(learning_rates):
        print("\nTesting for lr '{:02f}'...".format(lr))
        learned_optimizer(optlearner_models, epochs=TEST_EPOCHS, lr=lr, cb=epoch_pred_cbs[lr_num])

        # Reset the weights of the embedding layer
        #for layer_name, trainable in param_trainable.items():
        #    emb_layer = optlearner_model.get_layer(layer_name)
        #    layer_init_weights = initial_weights[:, int(layer_name[6:8])]
        #    emb_layer.set_weights(layer_init_weights.reshape(1, data_samples, 1))
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




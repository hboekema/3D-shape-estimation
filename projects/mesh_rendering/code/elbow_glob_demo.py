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

from callbacks import OptLearnerPredOnEpochEnd
from silhouette_generator import OptLearnerDataGenerator, OptLearnerExtraOutputDataGenerator
from optlearner import OptLearnerStaticArchitecture, OptLearnerStaticCosArchitecture, OptLearnerArchitecture, OptLearnerExtraOutputArchitecture, OptLearnerDistArchitecture, no_loss, false_loss
from smpl_np import SMPLModel
from render_mesh import Mesh

parser = ArgumentParser()
#parser.add_argument("--model", help="Path to the model weights")
parser.add_argument("--data", help="Path to the data directory")
args = parser.parse_args()

#info = {'model_weights_path': "../experiments/large_rot_result_2019-11-18_11:34:15/models/model.499-15.83.hdf5"}
#info = {'model_weights_path': "../experiments/2019-11-18_10:32:39/models/model.99-0.60.hdf5"}
#info = {'model_weights_path': "../experiments/2019-11-16_10:57:46/models/model.499-0.01.hdf5"}
# = {'model_weights_path': '../experiments/2019-11-13_11:38:42/models/model.399-0.00.hdf5'}
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("GPU used: |" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")

if args.data is None:
    args.data = "../data/train/"

np.random.seed(10)

# Experiment directory
#exp_dir = "/data/cvfs/hjhb2/projects/mesh_rendering/experiments/2019-12-14_15:27:34/"
#model_name = "model.2299-1689.94.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/mesh_rendering/experiments/2019-12-15_09:06:35/"
#model_name = "model.15499-561.11.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/mesh_rendering/experiments/2020-01-10_19:53:34/"
#model_name = "model.2400-107.61.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/mesh_rendering/experiments/2020-01-11_11:06:14/"
#model_name = "model.400-14.71.hdf5"
exp_dir ="/data/cvfs/hjhb2/projects/mesh_rendering/experiments/2020-01-11_15:44:21/"
model_name = "model.100-inf.hdf5"

model = exp_dir + "models/" + model_name
logs_dir = exp_dir + "logs/" + model_name + "/"
os.system('mkdir ' + logs_dir)
control_logs_dir = exp_dir + "logs/control/"
os.system('mkdir ' + control_logs_dir)
test_vis_dir = exp_dir + "test_vis/" + model_name + "/"
os.system('mkdir ' + test_vis_dir)
control_dir = exp_dir + "test_vis/" + "control/"
os.system('mkdir ' + control_dir)

# Generate the data from the SMPL parameters
smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
zero_params = np.zeros(shape=(85,))
zero_pc = smpl.set_params(beta=zero_params[72:82], pose=zero_params[0:72].reshape((24,3)), trans=zero_params[82:85])
#print("zero_pc: " + str(zero_pc))
base_params = 0.2 * (np.random.rand(85) - 0.5)
base_pc = smpl.set_params(beta=base_params[72:82], pose=base_params[0:72].reshape((24,3)), trans=base_params[82:85])


data_samples = 10000
X_indices = np.array([i for i in range(data_samples)])
X_params = 0.2 * np.random.rand(data_samples, 85)
#X_params = np.array([zero_params for i in range(data_samples)], dtype="float64")
#X_params[:, 0] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
X_params[:, 1] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_params[:, 2] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
X_params[:, 56] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_params[:, 57] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_params[:, 58] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
X_params[:, 59] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_params[:, 60] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_params[:, 61] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_pcs = np.array([zero_pc for i in range(data_samples)], dtype="float64")
#X_params = np.array([base_params for i in range(data_samples)], dtype="float32")
#X_pcs = np.array([base_pc for i in range(data_samples)], dtype="float32")

#X_params = 0.2 * np.random.rand(data_samples, 85)
X_pcs = []
for params in X_params:
    pc = smpl.set_params(beta=params[72:82], pose=params[0:72], trans=params[82:85])
    X_pcs.append(pc)
X_pcs = np.array(X_pcs)
X_data = [np.array(X_indices), np.array(X_params), np.array(X_pcs)]
Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85))]

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
def emb_init_weights(emb_params):
    def emb_init_wrapper(param, offset=False):
        def emb_init(shape, dtype="float32"):
            """ Initializer for the embedding layer """
            emb_params_ = emb_params[:, param]

            if offset:
                k = np.pi
                #k = 1.0
                offset_ = k * 2 * (np.random.rand(shape[0]) - 0.5)
                emb_params_[:] += offset_

            init = np.array(emb_params_, dtype=dtype).reshape(shape)
            #print("init shape: " + str(init.shape))
            #print("init values: " + str(init))
            #exit(1)
            return init
        return emb_init
    return emb_init_wrapper

emb_initialiser = emb_init_weights(X_params)
param_ids = ["param_{:02d}".format(i) for i in range(85)]
#trainable_params = ["param_00", "param_01", "param_02", "param_56", "param_57", "param_58", "param_59", "param_60", "param_61"]
trainable_params = ["param_01", "param_56", "param_59"]
param_trainable = { param: (param in trainable_params) for param in param_ids }

# Load model
#optlearner_inputs, optlearner_outputs = OptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, emb_size=data_samples)
optlearner_inputs, optlearner_outputs = OptLearnerStaticCosArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, emb_size=data_samples)
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
optlearner_model.compile(optimizer=optimizer, loss=[no_loss, false_loss, no_loss, false_loss,
                                                    false_loss,  # pc loss
                                                    false_loss,  # loss that updates smpl parameters
                                                    no_loss, no_loss, no_loss],
                                                loss_weights=[0.0,
                                                            1.0,  # delta_d loss weight
                                                            0.0,
                                                            1.0,  # pc loss weight
                                                            0.0,  # smpl loss weight
                                                            0.0, 0.0, 0.0, 0.0])
# Print model summary
optlearner_model.summary()


# Visualisation callback
epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=test_vis_dir, period=1, trainable_params=trainable_params, visualise=False)
epoch_pred_cb_control = OptLearnerPredOnEpochEnd(control_logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=control_dir, period=1, trainable_params=trainable_params,visualise=False)


def learned_optimizer(optlearner_model, epochs=50, lr=0.1):
    metrics_names = optlearner_model.metrics_names
    print("metrics_names: " + str(metrics_names))
    named_scores = {}
    epoch_pred_cb.set_model(optlearner_model)
    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1))
        print("----------------------")
        #print('emb layer weights'+str(emb_weights))
	#print('shape '+str(emb_weights[0].shape))
	y_pred = optlearner_model.predict(x_test)

        for emb_layer, param_num in trainable_layers.items():
	    emb_weights = emb_layer.get_weights()
            emb_weights += lr * np.array(y_pred[7][:, param_num]).reshape((data_samples, 1))
	    emb_layer.set_weights(emb_weights)

        # Evaluate model performance
        scores = optlearner_model.evaluate(x_test, y_test, batch_size=32)
        for i, score in enumerate(scores):
            named_scores[metrics_names[i]] = score
        #print("scores: " + str(scores))
        #exit(1)
        epoch_pred_cb.on_epoch_end(epoch=int(epoch), logs=named_scores)



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
    learned_optimizer(optlearner_model, lr=0.5)
    #regular_optimizer(optlearner_model, epochs=50)
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




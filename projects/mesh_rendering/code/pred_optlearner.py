import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import cv2
from keras.models import Model
from keras.optimizers import Adam
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
from datetime import datetime
import pickle

from callbacks import OptLearnerPredOnEpochEnd
from silhouette_generator import OptLearnerDataGenerator
from optlearner import OptLearnerArchitecture, OptLearnerDistArchitecture, OptLearnerSingleInputArchitecture, no_loss, false_loss
from smpl_np import SMPLModel
from render_mesh import Mesh

parser = ArgumentParser()
#parser.add_argument("--model", help="Path to the model weights")
parser.add_argument("--data", help="Path to the data directory")
args = parser.parse_args()

info = {'model_weights_path': "../experiments/opt_architecture_001/models/model.399-0.01.hdf5"}
# = {'model_weights_path': '../experiments/2019-11-13_11:38:42/models/model.399-0.00.hdf5'}
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
print("GPU used: |" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")

smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/./basicModel_f_lbs_10_207_0_v1.0.0.pkl')

if args.data is None:
    args.data = "../data/test/"

test_gen = OptLearnerDataGenerator(args.data, batch_size=10, pred_mode=True)

# Initialise the weights in the embedding layer
embedding_initializer = test_gen.yield_params(offset="right arm")


def
    # Load model
    optlearner_inputs, optlearner_outputs = OptLearnerArchitecture(embedding_initializer)
    #input_indices = [0, 2]
    #output_indices = [0, 2, 3, 5]
    #optlearner_model = Model(inputs=[input_ for i, input_ in enumerate(optlearner_inputs) if i in input_indices], outputs=[output for i, output in enumerate(optlearner_outputs) if i in output_indices])
    optlearner_model = Model(inputs=optlearner_inputs, outputs=optlearner_outputs)
    optlearner_model.load_weights(info["model_weights_path"])

# Load model
#optlearner_inputs, optlearner_outputs = OptLearnerArchitecture(embedding_initializer)
#input_indices = [0, 2]
#output_indices = [0, 2, 3, 5]
#optlearner_model = Model(inputs=[input_ for i, input_ in enumerate(optlearner_inputs) if i in input_indices], outputs=[output for i, output in enumerate(optlearner_outputs) if i in output_indices])
#optlearner_model = Model(inputs=optlearner_inputs, outputs=optlearner_outputs)
#optlearner_model.load_weights(info["model_weights_path"])

# Freeze all layers except for the embedding layer
trainable_layers = ["parameter_embedding"]
for layer in optlearner_model.layers:
    if layer.name not in trainable_layers:
        layer.trainable = False

# Set the weights of the embedding layer
#embedding_layer = optlearner_model.get_layer("parameter_embedding")
#print(np.array(embedding_layer.get_weights()).shape)
#embedding_layer.set_weights(embedding_initializer(shape=(1, 1000, 85), dtype="float32"))

# Compile the model
learning_rate = 0.01
optlearner_model.compile(optimizer=Adam(lr=learning_rate, decay=0.0), loss=[no_loss, false_loss, no_loss, false_loss, false_loss, false_loss],
        loss_weights=[0.0, 0.0, 0.0, 1.0, 0.0, 100.0])
optlearner_model.summary()

# Create experiment directory
exp_dir = "../experiments/demo_test" + str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + "/"
logs_dir = exp_dir + "logs/"
test_vis_dir = exp_dir + "test_vis/"
os.mkdir(exp_dir)
os.mkdir(test_vis_dir)
os.mkdir(logs_dir)

# Visualisation input
test_ids = os.listdir(args.data)[:10]
test_sample_indices = []
test_sample_offset_params = []
test_sample_params = []
test_sample_pcs = []
test_sample_silh = []
for i, sample in enumerate(test_ids):
    with open(os.path.join(args.data, sample), 'r') as handle:
        data_dict = pickle.load(handle)
    test_sample_indices.append(i)
    test_sample_params.append(data_dict["parameters"])
    params = data_dict["parameters"]
    params[57:60] = np.random.rand(3) # change left elbow
    test_sample_offset_params.append(params)
    test_sample_pcs.append(data_dict["pointcloud"])
    test_sample_silh.append(data_dict["silhouette"])

test_sample_input = [np.array(test_sample_offset_params), np.array(test_sample_params), np.array(test_sample_pcs)]
test_sample_output = [np.array(test_sample_params), np.zeros(shape=(self.batch_size,)), np.array(test_sample_pcs), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size, 85))]
# Visualisation callback
#epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, test_inputs=test_sample_input, test_silh=test_sample_silh,
#        pred_path=test_vis_dir, period=1, visualise=False)

#X, Y = test_gen.__getitem__(0)

# Train the model
#optlearner_model.fit(
#        X,
#        Y,
#        epochs=100,
#        callbacks=[epoch_pred_cb]
#        )

# Evaluate the model performance
#eval_log = optlearner_model.evaluate_generator(test_gen)

#eval_string = ""
#for i in range(len(eval_log)):
#    eval_string += str(optlearner_model.metrics_names[i]) + ": " + str(eval_log[i]) + "  "
#print(eval_string)

# Predict the parameters from the silhouette and generate a mesh
for
    prediction = optlearner_model.predict(test_sample_input)
for i, pred in enumerate(prediction, 1):


    learned_pc = pred[2]

    gt_silhoutte = test_sample_silh[i-1]
    pred_silhouette = Mesh(pointcloud=learned_pc).render_silhouette(show=False)

    diff_silh = abs(gt_silhouette - pred_silhouette)
    #print(diff_silh.shape)
    #cv2.imshow("Diff silh", diff_silh)
    #cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:03d}.diff_silh_{:03d}.png".format(data_type, epoch + 1, i)), diff_silh.astype("uint8"))
    silh_comp = np.concatenate([gt_silhouette, pred_silhouette, diff_silh])
    cv2.imwrite(os.path.join(test_vis_dir, "{}_epoch.{:05d}.silh_comp_{:03d}.png".format(data_type, epoch + 1, i)), silh_comp.astype("uint8"))


#pred_params = prediction[0]
#real_params = Y_test[0]
#
#for pred in pred_params:
#    pointcloud = smpl.set_params(pred[10:82], pred[0:10], pred[82:85])
#    pred_silhouette = Mesh(pointcloud=pointcloud).render_silhouette()
#
#    pointcloud = smpl.set_params(real_params[10:82], real_params[0:10], real_params[82:85])
#    real_silhouette = Mesh(pointcloud=pointcloud).render_silhouette()
#
#    diff_silhouette = abs(real_silhouette - pred_silhouette)
#
#    comp_silh = np.concatenate([real_silhouette, pred_silhouette, diff_silhouette])
#    plt.imshow(comp_silh, cmap='gray')
#    plt.show()




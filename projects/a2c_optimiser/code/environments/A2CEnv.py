
import sys

import numpy as np
from keras import Model
from keras.layers import
from copy import copy

sys.path.append('/data/cvfs/hjhb2/projects/a2c_optimiser/code/')
#sys.path.append('/data/cvfs/hjhb2/projects/a2c_optimiser/code/keras_rotationnet_v2_demo_for_hidde/')
from architectures.architecture_helpers import get_pc, load_smpl_params, get_sin_metric


class A2CEnv:
    """ Environment for A2C pose estimation task """
    def __init__(self, TARGET_OFFSET, TARGET_PARAMS_TO_OFFSET, PARAM_OFFSET, PARAMS_TO_OFFSET, SMPL, BATCH_SIZE=1, opt_lr=0.05, opt_iter=5, reward_factor=100, epsilon=1e-3):
        # Environment configuration parameters
        self.POSE_OFFSET = POSE_OFFSET
        self.PARAM_OFFSET = PARAM_OFFSET
        self.smpl = SMPL
        self.BATCH_SIZE = BATCH_SIZE

        # Parameters for batched TF implementation of SMPL model
        self.smpl_params, self.input_info, self.faces = load_smpl_params()

        # Reward configuration parameters
        self.opt_lr = opt_lr
        self.opt_iter = opt_iter
        self.reward_factor = reward_factor
        self.epsilon = epsilon
        self.GD_model = self.init_GD_model()

        # Store environment states etc.
        self.target_params, self.target_pcs = self.generate_targets()
        self.pred_params, self.pred_pcs = self.generate_preds()


    # Core methods
    # ------------

    def reset(self):
        """ Resets the environment and yields a random state """
        # Generate new random examples and reset the gradient descent model
        self.target_params, self.target_pcs = self.generate_targets()
        self.pred_params, self.pred_pcs = self.generate_preds()
        self.GD_model = self.init_GD_model()

        state = self.target_pcs - self.pred_pcs

        return state

    def step(self, action):
        """ Action is the SMPL parameter difference to apply to current estimate and must be shape (batch_size, 85) """
        next_state = self.update_state(action)
        reward, done = self.calculate_reward()

        return next_state, reward, done

    def update_state(self, action):
        """ Update the state with the action """
        # Update the parameters and PCs, then take the difference between the target and predicted PCs
        pred_params, pred_pcs = update_params(action)
        pc_difference = self.target_pcs - pred_pcs

        return pc_difference

    def update_params(self, param_diff):
        """ Update the SMPL parameter estimate and render the PCs """
        # Update the predicted parameters
        self.pred_params += param_diff

        # Use the SMPL model to render these parameters to point clouds
        pred_pcs = np.zeros((self.BATCH_SIZE, 6890, 3))
        for i, params in enumerate(pred_params):
            pred_pcs[i] = self.smpl.set_params(beta=params[72:82], pose=params[0:72].reshape((24,3)), trans=params[82:85])

        self.pred_pcs = pred_pcs

        return self.pred_params, self.pred_pcs

    def calculate_reward(self):
        """ Return the reward for this state - the reward is given by the Euclidean loss of the PCs after x gradient descent iterations """
        emb_indices = np.array([i for i in range(self.BATCH_SIZE)])

        self.GD_model.fit(
                x=[emb_indices]
                y=[self.target_params, self.target_pcs]
                batch_size=self.BATCH_SIZE,
                epochs=self.opt_iter,
                verbose=0
                )

        # Get the new predicted parameters and compute the MSE
        new_pred_params = self.GD_model.get_layer("pred_embedding").get_weights()
        params_mse = (np.square(self.target_params - new_pred_params)).mean(axis=1)

        # Convert the MSE into a reward
        reward = np.mean(self.reward_factor * np.exp(-params_mse))
        done = [mse <= self.epsilon for mse in params_mse]
        if self.BATCH_SIZE == 1:
            done = done[0]

        return reward, done


    # Initialisation methods
    # ----------------------

    def init_GD_model(self):
        """ Build a Keras Functional Model with parameter embeddings for the gradient descent reward network """
        def emb_init_wrapper(target_params):
            def emb_init(shape, dtype=None):
                return target_params
            return emb_init

        # Network inputs
        emb_index = Input(shape=(1,), name="pred_emb_index")

        # Retrieve the predicted parameters from the Embedding layer
        pred_embedding = Embedding(self.BATCH_SIZE, 85, name="pred_embedding", trainable=True, embeddings_initializer=emb_init_wrapper(self.pred_params))(emb_index)
        pred_params = Reshape(target_shape=(85,), name="pred_params")(pred_embedding)

        # Render the predicted point clouds
        pred_pc = get_pc(pred_params, self.smpl_params, self.input_info, self.faces)
        print("pred_pc shape: " + str(pred_pc.shape))

        # Instantiate Model instance with the chosen topology
        GD_model = Model(inputs=[emb_index], outputs=[pred_params, pred_pc])
        #GD_model.summary()

        # Define optimiser and losses and compile the model
        optimizer = Adam(lr=self.opt_lr, decay=0.0)
        losses = ["mse", "mse"]
        loss_weights = [1.0, 1.0]
        metrics = {"pred_params": get_sin_metric}
        GD_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

        self.GD_model = GD_model

        return GD_model

    def generate_targets(self):
        """ Create target params within the permissible range and render corresponding PCs """
        # Generate the GT parameters first
        target_params = np.zeros((self.BATCH_SIZE, 85))
        target_params = offset_params(target_params, self.POSE_OFFSET)

        # Use the SMPL model to render these parameters to point clouds
        target_pcs = np.zeros((self.BATCH_SIZE, 6890, 3))
        for i, params in enumerate(target_params):
            target_pcs[i] = self.smpl.set_params(beta=params[72:82], pose=params[0:72].reshape((24,3)), trans=params[82:85])

        self.target_params = target_params
        self.target_pcs = target_pcs

        return target_params, target_pcs

    def generate_preds(self):
        """ Create initial predictions which are offset from the targets by the desired amount """
        # Disturb the target params by the chosen amount
        target_params = copy(self.params)
        pred_params = offset_params(target_params, self.PARAM_OFFSET)

        # Render the 'predicted' parameters to point clouds
        pred_pcs = np.zeros((self.BATCH_SIZE, 6890, 3))
        for i, params in enumerate(pred_params):
            pred_pcs[i] = self.smpl.set_params(beta=params[72:82], pose=params[0:72].reshape((24,3)), trans=params[82:85])


        self.pred_params = pred_params
        self.pred_pcs = pred_pcs

        return pred_params, pred_pcs


    @staticmethod
    def format_offset_dict(k, params):
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


    @staticmethod
    def offset_params(params, offsets):
        """ Apply offset to the chosen parameters in params """
        sorted_param_offsets = sorted(offsets.keys(), key=lambda x: int(x[6:8]))
        offset_params_int = [int(param[6:8]) for param in sorted_param_offsets]
        data_samples = params.shape[0]
        params[:, offset_params_int] = np.array([offsets[param] * (1 - 2*np.random.rand(data_samples)) for param in sorted_param_offsets]).T

        return params



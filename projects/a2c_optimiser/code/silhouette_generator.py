import os
import numpy as np
import tensorflow as tf
import keras
import pickle

from smpl_np import SMPLModel
from render_mesh import Mesh
from generate_data import gen_data


class OptLearnerUpdateGenerator(keras.utils.Sequence):
    def __init__(self, num_samples, reset_period, POSE_OFFSET, PARAMS_TO_OFFSET, batch_size=32, smpl=None, shuffle=True, save_path="./cb_samples.npz"):
        self.num_samples = num_samples
        self.reset_period = reset_period
        if isinstance(POSE_OFFSET, (int, float)):
            k = {param: POSE_OFFSET for param in PARAMS_TO_OFFSET}
        else:
            # k must be a dict with an entry for each variable parameter
            k = POSE_OFFSET
        self.k = k
        self.PARAMS_TO_OFFSET = PARAMS_TO_OFFSET
        self.batch_size = batch_size
        if smpl is None:
            smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        self.smpl = smpl
        self.shuffle = shuffle
        self.params, self.pcs = gen_data(POSE_OFFSET, PARAMS_TO_OFFSET, self.smpl, data_samples=num_samples, save_dir=None, render_silhouette=False)
        self.indices = np.array([i for i in range(num_samples)])
        self.save_path = save_path
        #print("generator params shape at init: " + str(self.params.shape))
        #print("generator pcs shape at init: " + str(self.pcs.shape))
        print("generator params first entry: " + str(self.params[0]))
        print("generator params final entry: " + str(self.params[-1]))

        # Hard-coded parameters
        self.epoch = 0
        self.num_examples = 5
        self.cb_samples = np.linspace(0, self.num_samples, num=self.num_examples, dtype=int)
        self.cb_samples[-1] -= 1
        print("samples: " + str(self.cb_samples))

        # Store initial data
        if self.save_path is not None:
            with open(self.save_path, 'w') as f:
                np.savez(f, indices=self.indices[self.cb_samples], params=self.params[self.cb_samples], pcs=self.pcs[self.cb_samples])

    def on_epoch_end(self):
        # NOTE - this function is NOT thread-safe
        #print("\nEPOCH: " + str(self.epoch) + "\n")
        # Distract selected base poses
        # Update a block of parameters
        #print("Distracting base pose...")
        if self.epoch >= 0:
            BL_SIZE = self.num_samples // self.reset_period
            print("epoch: " + str(self.epoch))
            BL_INDEX = self.epoch % self.reset_period
            print("BL_SIZE: " + str(BL_SIZE))
            print("BL_INDEX: " + str(BL_INDEX))

            params_to_offset = [key for key in self.k.keys() if key in self.PARAMS_TO_OFFSET]
            sorted_keys = sorted(params_to_offset, key=lambda x: int(x[6:8]))
            sorted_keys_num = [int(x[6:8]) for x in sorted_keys]
            #print(sorted_keys)
            sorted_offsets = [self.k[key] for key in sorted_keys]

            weights_new = np.array(1-2*np.random.rand(BL_SIZE, len(sorted_keys_num))) * sorted_offsets
            #print("weights_new shape: " + str(weights_new.shape))
            #print(BL_INDEX*BL_SIZE)
            #print((BL_INDEX+1)*BL_SIZE)
            self.params[BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE, sorted_keys_num] = weights_new    # only update the required block of weights

            # Now render the point clouds for the new weights
            updated_params = self.params[BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE]
            for i, params in enumerate(updated_params, BL_INDEX*BL_SIZE):
                updated_pc = self.smpl.set_params(beta=params[72:82], pose=params[0:72], trans=params[82:85])
                self.pcs[i] = updated_pc

            # Finally, shuffle the data
            if self.shuffle:
                np.random.shuffle(self.indices)

            # Write the examples to file
            if self.save_path is not None:
                with open(self.save_path, 'w') as f:
                    np.savez(f, indices=self.cb_samples, params=self.params[self.cb_samples], pcs=self.pcs[self.cb_samples])

            #print("generator params shape after change: " + str(self.params.shape))
            #print("generator pcs shape after change: " + str(self.pcs.shape))

        #print("Finished.")
        self.epoch += 1

    def __len__(self):
        return int(self.num_samples // self.batch_size)

    def __getitem__(self, index):
        """ Get the next batch (by index) """
        # Ensure that the item index is always valid
        index = index % self.__len__()

        X_batch_index = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch_params = self.params[X_batch_index]
        X_batch_pc = self.pcs[X_batch_index]

        # TODO: adjust for different architectures
        X_batch = [np.array(X_batch_index), np.array(X_batch_params), np.array(X_batch_pc)]
        Y_batch = [np.zeros((self.batch_size, 85)), np.zeros((self.batch_size,)), np.zeros((self.batch_size, 6890, 3)), np.zeros((self.batch_size,)), np.zeros((self.batch_size,)), np.zeros((self.batch_size, 85)), np.zeros((self.batch_size, 85)), np.zeros((self.batch_size, 85)), np.zeros((self.batch_size, 85)), np.zeros((self.batch_size, 31))]

        return X_batch, Y_batch

    def yield_data(self):
        """ Yield all of the data, correctly ordered """
        ordered_indices = [i for i in range(self.num_samples)]
        X_data = [np.array(ordered_indices), np.array(self.params), np.array(self.pcs)]
        Y_data = [np.zeros((self.num_samples, 85)), np.zeros((self.num_samples,)), np.zeros((self.num_samples, 6890, 3)), np.zeros((self.num_samples,)), np.zeros((self.num_samples,)), np.zeros((self.num_samples, 85)), np.zeros((self.num_samples, 85)), np.zeros((self.num_samples, 85)), np.zeros((self.num_samples, 85)), np.zeros((self.num_samples, 31))]

        return X_data, Y_data


class OptLearnerExtraOutputDataGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, pred_mode=False, debug=False, smpl=None):
        self.data_dir = data_dir
        ids = os.listdir(self.data_dir)
        self.ids = [(i, id_) for i, id_ in enumerate(ids)]
        self.ordered_ids = self.ids
        self.batch_size = batch_size
        self.pred_mode = pred_mode
        self.debug = debug
        self.smpl = smpl
        self.zero_pc = smpl.set_params(beta=np.zeros(10), pose=np.zeros((24,3)), trans=np.zeros(3))

    def on_epoch_end(self):
        """ Shuffle data ids to load in next epoch """
        if not self.debug:
            np.random.shuffle(self.ids)

    def __len__(self):
        if self.debug:
            # Always output the same batch
            return 2
            #return 1
        else:
            #print("len of generator: " + str(int(np.ceil(len(self.ids) / self.batch_size))))
            return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """ Get the next batch (by index) """
        X_batch_index = []
        X_batch_params = []
        X_batch_pc = []

        # Ensure that the item index is always valid
        index = index % self.__len__()
        #print("Index: " + str(index))
        for i in range(index*self.batch_size, (index + 1)*self.batch_size):
            curr_index = self.ids[i][0]
            curr_id = self.ids[i][1]

            X_batch_index.append(curr_index)

            with open(os.path.join(self.data_dir, curr_id), 'r') as handle:
                data_dict = pickle.load(handle)

            parameters = data_dict["parameters"]
            X_batch_params.append(parameters)
            X_batch_pc.append(data_dict["pointcloud"])

        if not self.pred_mode:
            X_batch = [np.array(X_batch_index), np.array(X_batch_params), np.array(X_batch_pc)]
            #gt_pcs = np.array([self.zero_pc for i in range(self.batch_size)])
            #X_batch = [np.array(X_batch_index), np.zeros(shape=(self.batch_size, 85)), np.array(gt_pcs)]
            Y_batch = [np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size, 6890, 3)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size, 85))]
        else:
            X_batch = [np.array(X_batch_index), np.zeros(shape=(self.batch_size, 85)), np.array(X_batch_pc)]
            Y_batch = [np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size, 6890, 3)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size, 85))]

        return X_batch, Y_batch

    def yield_params(self, offset=None):
        """ Wrapper function for yielding all of the parameters in the directory """
        def yield_params_wrapped(shape, dtype="float32"):
            X_batch_params = []
            for i in range(len(self.ordered_ids)):
                curr_id = self.ordered_ids[i][1]

                #with open(os.path.join(self.data_dir, curr_id), 'r') as handle:
                #    data_dict = pickle.load(handle)
                parameters = np.zeros((85,))

            #    parameters = data_dict["parameters"]
                if offset is not None:
                    offset_ = offset
                    if offset_ == "right arm":
                        np.random.seed(10)
                        offset_ = np.zeros((85,))
                        #offset_[42:45] = np.random.rand(3) # change left shoulder 1
                        #offset_[51:54] = np.random.rand(3) # change left shoulder 2
                        offset_[57:60] = 0.2 * np.random.randint(low=-1, high=1)     # change left elbow
                        #offset_[63:66] = np.random.rand(3) # change left wrist
                        #offset_[69:72] = np.random.rand(3) # change left fingers
                        #offset_[72:75] = np.random.rand(3) # change global translation

                    parameters += offset_

                X_batch_params.append(parameters)
            return np.array(X_batch_params, dtype=dtype).reshape(shape)

        return yield_params_wrapped


class OLExtraOutputZeroPoseDataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=32, pred_mode=False, smpl=None):
        self.batch_size = batch_size
        self.pred_mode = pred_mode
        self.smpl = smpl
        self.zero_pc = smpl.set_params(beta=np.zeros(10), pose=np.zeros(72), trans=np.zeros(3))
        self.parameters = self.gen_parameters()

    def on_epoch_end(self):
        """ Shuffle data ids to load in next epoch """
        pass

    def __len__(self):
        return int(np.ceil(len(self.parameters)/self.batch_size))

    def __getitem__(self, index):
        """ Get the next batch (by index) """
        index = index % self.__len__()
        X_batch_index = [i for i in range(index*self.batch_size, (index+1)*self.batch_size)]

        gt_pc = np.array([self.zero_pc for i in range(self.batch_size)])

        X_batch = [np.array(X_batch_index), np.zeros(shape=(self.batch_size, 85)), gt_pc]
        Y_batch = [np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size, 6890, 3)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size, 85))]

        return X_batch, Y_batch

    def gen_parameters(self, offset=True):
        parameters = np.zeros(shape=(1000, 85))

        offset_ = np.zeros((1000, 85))
        if offset is True:
            np.random.seed(10)
            #offset_[42:45] = np.random.rand(3) # change left shoulder 1
            #offset_[51:54] = np.random.rand(3) # change left shoulder 2
            offset_[:, 57] = 0.2 * np.random.randint(low=-1, high=1, size=(1000,))     # change left elbow
            #offset_[63:66] = np.random.rand(3) # change left wrist
            #offset_[69:72] = np.random.rand(3) # change left fingers
            #offset_[72:75] = np.random.rand(3) # change global translation

        parameters += offset_
        self.parameters = parameters
        #print(parameters[0:5, 54:60])
        #exit(1)
        return parameters

    def yield_params(self, offset=True):
        """ Wrapper function for yielding all of the parameters in the directory """
        def yield_params_wrapped(shape, dtype="float32"):
            if self.parameters is None:
                self.gen_parameters(offset)
            #print(self.parameters[0:5, 54:60])
            #exit(1)
            return np.array(self.parameters, dtype=dtype).reshape(shape)

        return yield_params_wrapped


class OptLearnerDataGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, pred_mode=False, debug=False):
        self.data_dir = data_dir
        ids = os.listdir(self.data_dir)
        self.ids = [(i, id_) for i, id_ in enumerate(ids)]
        self.ordered_ids = self.ids
        self.batch_size = batch_size
        self.pred_mode = pred_mode
        self.debug = debug

    def on_epoch_end(self):
        """ Shuffle data ids to load in next epoch """
        if not self.debug:
            np.random.shuffle(self.ids)

    def __len__(self):
        if self.debug:
            # Always output the same batch
            return 1
        else:
            #print("len of generator: " + str(int(np.ceil(len(self.ids) / self.batch_size))))
            return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """ Get the next batch (by index) """
        X_batch_index = []
        X_batch_params = []
        X_batch_pc = []

        # Ensure that the item index is always valid
        index = index % self.__len__()
        #print("Index: " + str(index))
        for i in range(index*self.batch_size, (index + 1)*self.batch_size):
            curr_index = self.ids[i][0]
            curr_id = self.ids[i][1]

            X_batch_index.append(curr_index)

            with open(os.path.join(self.data_dir, curr_id), 'r') as handle:
                data_dict = pickle.load(handle)

            X_batch_params.append(data_dict["parameters"])
            X_batch_pc.append(data_dict["pointcloud"])

        if not self.pred_mode:
            X_batch = [np.array(X_batch_index), np.array(X_batch_params), np.array(X_batch_pc)]
            Y_batch = [np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size, 6890, 3)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size, 85))]
        else:
            X_batch = [np.array(X_batch_index), np.zeros(shape=(self.batch_size, 85)), np.array(X_batch_pc)]
            Y_batch = [np.array(X_batch_params), np.zeros(shape=(self.batch_size,)), np.array(X_batch_pc), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size, 85))]

        return X_batch, Y_batch

    def yield_params(self, offset=None):
        """ Wrapper function for yielding all of the parameters in the directory """
        def yield_params_wrapped(shape, dtype="float32"):
            X_batch_params = []
            for i in range(len(self.ordered_ids)):
                curr_id = self.ordered_ids[i][1]

                with open(os.path.join(self.data_dir, curr_id), 'r') as handle:
                    data_dict = pickle.load(handle)

                parameters = data_dict["parameters"]
                if offset is not None:
                    offset_ = offset
                    if offset_ == "right arm":
                        np.random.seed(10)
                        offset_ = np.zeros(85)
                        offset_[42:45] = np.random.rand(3) # change left shoulder 1
                        offset_[51:54] = np.random.rand(3) # change left shoulder 2
                        offset_[57:60] = np.random.rand(3) # change left elbow
                        offset_[63:66] = np.random.rand(3) # change left wrist
                        offset_[69:72] = np.random.rand(3) # change left fingers
                        #offset_[72:75] = np.random.rand(3) # change global translation

                    parameters += offset_

                X_batch_params.append(parameters)
            return np.array(X_batch_params, dtype=dtype).reshape(shape)

        return yield_params_wrapped


class NormLearnerDataGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, debug=False):
        self.data_dir = data_dir
        ids = os.listdir(self.data_dir)
        self.ids = [(i, id_) for i, id_ in enumerate(ids)]
        self.ordered_ids = self.ids
        self.batch_size = batch_size
        self.debug = debug

    def on_epoch_end(self):
        """ Shuffle data ids to load in next epoch """
        if not self.debug:
            self.ids = np.random.shuffle(self.ids)

    def __len__(self):
        if self.debug:
            # Always output the same batch
            return 1
        else:
            return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """ Ge tthe next batch (by index) """
        X_batch_index = []
        X_batch_params = []
        X_batch_pc = []

        # Ensure that the item index is always valid
        index = index % self.__len__()
        for i in range(index*self.batch_size, (index + 1)*self.batch_size):
            curr_index = self.ids[i][0]
            curr_id = self.ids[i][1]

            X_batch_index.append(curr_index)

            with open(os.path.join(self.data_dir, curr_id), 'r') as handle:
                data_dict = pickle.load(handle)

            X_batch_params.append(data_dict["parameters"])
            X_batch_pc.append(data_dict["pointcloud"])

        X_batch = [np.array(X_batch_index), np.array(X_batch_params), np.array(X_batch_pc)]
        Y_batch = [np.zeros(shape=(self.batch_size, 85)), np.zeros(shape=(self.batch_size,)), np.zeros(shape=(self.batch_size, 6890, 3)), np.zeros(shape=(self.batch_size,))]

        return X_batch, Y_batch

    def yield_params(self, offset=None):
        def yield_params_wrapped(shape, dtype="float32"):
            X_batch_params = []

            for i in range(len(self.ordered_ids)):
                curr_id = self.ordered_ids[i][1]

                with open(os.path.join(self.data_dir, curr_id), 'r') as handle:
                    data_dict = pickle.load(handle)

                parameters = data_dict["parameters"]
                if offset is not None:
                    parameters += offset

                X_batch_params.append(parameters)

            return np.array(X_batch_params, dtype=dtype)

        return yield_params_wrapped


class LoadedDataGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32):#, img_dim=(256, 256), frac_randomised=0.2, noise=0.01, debug=False):
        self.data_dir = data_dir
        self.ids = os.listdir(self.data_dir)
        self.batch_size = batch_size
        #self.img_dim = img_dim
        #self.frac_randomised = frac_randomised  # fraction of parameters to generate randomly in each batch
        #self.noise = noise
        #self.smpl = smpl
        #self.debug = debug
        #self.debug_X = None
        #self.debug_Y = None

    def on_epoch_end(self):
        """ Shuffle data ids to load in next epoch """
        self.ids = np.random.shuffle(self.ids)

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        X_batch = []
        Y_batch_params = []
        Y_batch_pc = []
        for i in range(self.batch_size):
            curr_id = self.ids[i]

            with open(os.path.join(data_dir, curr_id), 'r') as handle:
                data_dict = pickle.load(handle)

            X_batch.append(data_dict["silhouette"])
            Y_batch_params.append(data_dict["parameters"])
            Y_batch_pc.append(data_dict["pointcloud"])

        Y_batch = [np.array(Y_batch_params), np.array(Y_batch_pc)]

        return X_batch, Y_batch


class SilhouetteDataGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, smpl, batch_size=32, img_dim=(256, 256), frac_randomised=0.2, noise=0.01, debug=False):
        self.data_dir = data_dir
        self.ids = None
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.frac_randomised = frac_randomised  # fraction of parameters to generate randomly in each batch
        self.noise = noise
        self.smpl = smpl
        self.debug = debug
        self.debug_X = None
        self.debug_Y = None

    def on_epoch_end(self):
        """ Shuffle data ids to load in next """
        self.ids = os.listdir(self.data_dir)
        self.ids = np.shuffle(self.ids)

    def __len__(self):
        return int(np.ceil(len(os.listdir(self.data_dir)) / self.batch_size))

#    def __next__(self, *args, **kwargs):
#       return self.next(*args, **kwargs)
#
#    def next(self):
#        """For python 2.x
#        # Returns the next batch.    """
#        return __getitem__(0)

    def __getitem__(self, item):
        """ Yield batches of data """
        # Load the SMPL model
        #smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/./basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        smpl = self.smpl

        if self.debug:
            if self.debug_X is None:
                Y_batch_params = []
                Y_batch_pc = []
                X_batch = []
                for i in range(self.batch_size):
                    # Generate artificial data
                    pose = 0.65 * (np.random.rand(smpl.pose_shape[0], smpl.pose_shape[1]) - 0.5)
                    beta = 0.2 * (np.random.rand(smpl.beta_shape[0]) - 0.5)
                    trans = np.zeros(smpl.trans_shape[0])
                    #trans = 0.1 * (np.random.rand(smpl.trans_shape[0]) - 0.5)

                    # Create the body mesh
                    pointcloud = smpl.set_params(beta=beta, pose=pose, trans=trans)
                    Y_batch_params.append(np.concatenate([pose.ravel(), beta, trans]))
                    Y_batch_pc.append(pointcloud)

                    # Render the silhouette
                    silhouette = Mesh(pointcloud=pointcloud).render_silhouette(dim=self.img_dim, show=False)
                    X_batch.append(np.array(silhouette))

                # Preprocess the batches and yield them
                Y_batch = [np.array(Y_batch_params), np.array(Y_batch_pc)]
                X_batch = np.array(X_batch, dtype="float32")
                X_batch /= 255
                X_batch = X_batch.reshape((X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1))

                self.debug_X = X_batch
                self.debug_Y = Y_batch

            else:
                X_batch = self.debug_X
                Y_batch = self.debug_Y

        else:
            # Split of random and real data
            num_artificial = int(np.round(self.frac_randomised * self.batch_size))
            num_real = int(self.batch_size - num_artificial)

            # Retrieve a random batch of parameters from the data directory
            if num_real > 0:
                data = np.array(os.listdir(self.data_dir))
                Y_batch_ids = data[np.random.randint(low=0, high=data.shape[0], size=num_real)]
            else:
                Y_batch_ids = []

            Y_batch_params = []
            Y_batch_pc = []
            X_batch = []
            for Y_id in Y_batch_ids:
                # Fetch the real data
                Y = np.load(os.path.join(self.data_dir, Y_id))

                # Add a small amount of noise to the data
                Y += np.random.uniform(low=-self.noise, high=self.noise, size=Y.shape)
                Y_batch_params.append(Y)

                # Now generate the silhouette from the SMPL meshes
                # Create the body mesh
                pose = Y[:72]
                beta = Y[72:82]
                trans = Y[82:]
                pointcloud = smpl.set_params(pose.reshape((24, 3)), beta, trans)

                # Render the silhouette
                silhouette = Mesh(pointcloud=pointcloud).render_silhouette(dim=self.img_dim, show=False)
                X_batch.append(np.array(silhouette))

            for i in range(num_artificial):
                # Generate artificial data
                pose = 0.65 * (np.random.rand(smpl.pose_shape[0], smpl.pose_shape[1]) - 0.5)
                beta = 0.2 * (np.random.rand(smpl.beta_shape[0]) - 0.5)
                trans = np.zeros(smpl.trans_shape[0])
                #trans = 0.1 * (np.random.rand(smpl.trans_shape[0]) - 0.5)

                # Create the body mesh
                pointcloud = smpl.set_params(beta=beta, pose=pose, trans=trans)
                Y_batch_params.append(np.concatenate([pose.ravel(), beta, trans]))
                Y_batch_pc.append(pointcloud)

                # Render the silhouette
                silhouette = Mesh(pointcloud=pointcloud).render_silhouette(dim=self.img_dim, show=False)
                X_batch.append(np.array(silhouette))

            # Preprocess the batches and yield them
            Y_batch = [np.array(Y_batch_params), np.array(Y_batch_pc)]
            X_batch = np.array(X_batch, dtype="float32")
            X_batch /= 255
            X_batch = X_batch.reshape((X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1))

        #print("X_batch shape " + str(X_batch.shape))
        #print("Y_batch shape " + str(Y_batch.shape))
        #X_batch = list(X_batch)

        return X_batch, Y_batch



if __name__ == "__main__":
    smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/./basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    data_gen = SilhouetteDataGenerator("../../../data/AMASS/train/", smpl, batch_size=16, img_dim=(256, 256), frac_randomised=1.0)
    X_batch, Y_batch = data_gen.__getitem__(0)

    print(X_batch.shape)

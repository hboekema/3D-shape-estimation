
import numpy as np
from tqdm import tqdm
#import cv2
from render_mesh import Mesh
from rotation_helpers import rodrigues_to_euler


def gen_data(POSE_OFFSET, PARAMS_TO_OFFSET, smpl, data_samples=10000, save_dir=None, render_silhouette=True):
    """ Generate random body poses """
    POSE_OFFSET = format_distractor_dict(POSE_OFFSET, PARAMS_TO_OFFSET)

    zero_params = np.zeros(shape=(85,))
    zero_pc = smpl.set_params(beta=zero_params[72:82], pose=zero_params[0:72].reshape((24,3)), trans=zero_params[82:85])
    #print("zero_pc: " + str(zero_pc))

    # Generate and format the data
    X_indices = np.array([i for i in range(data_samples)])
    X_params = np.array([zero_params for i in range(data_samples)], dtype="float32")
    if not all(value == 0.0 for value in POSE_OFFSET.values()):
        X_params = offset_params(X_params, PARAMS_TO_OFFSET, POSE_OFFSET)
        X_pcs = np.array([smpl.set_params(beta=params[72:82], pose=params[0:72], trans=params[82:85]) for params in X_params])
    else:
        X_pcs = np.array([zero_pc for i in range(data_samples)], dtype="float32")

    if render_silhouette:
        X_silh = []
        print("Generating silhouettes...")
        for pc in tqdm(X_pcs):
            # Render the silhouette from the point cloud
            silh = Mesh(pointcloud=pc).render_silhouette(show=False)
            X_silh.append(silh)

        X_silh = np.array(X_silh)
        print("Finished generating data.")

    if save_dir is not None:
        # Save the generated data in the given location
        print("Saving generated samples...")
        for i in tqdm(range(data_samples)):
            sample_id = "sample_{:05d}".format(i+1)
            if render_silhouette:
                np.savez(save_dir + sample_id + ".npz", smpl_params=X_params[i], pointcloud=X_pcs[i], silhouette=X_silh[i])
            else:
                np.savez(save_dir + sample_id + ".npz", smpl_params=X_params[i], pointcloud=X_pcs[i], silhouette=X_silh[i])

        print("Finished saving.")

    if render_silhouette:
        return X_params, X_pcs, X_silh
    else:
        return X_params, X_pcs


def load_data(load_dir, num_samples=10000, load_silhouettes=False):
    """ Load previously generated data """
#    param_dir = load_dir + "smpl_params/"
#    pc_dir = load_dir + "pointclouds/"
#    silh_dir = load_dir + "silhouettes/"

    X_params = []
    X_pcs = []
    X_silh = []
    print("Loading data from '{}'...".format(load_dir))
    for i in tqdm(range(num_samples)):
        sample_id = "sample_{:05d}".format(i+1)

#        params = np.loadtxt(param_dir + sample_id + ".csv")
#        pc = Mesh(filepath=pc_dir + sample_id + ".obj").verts
#        if load_silhouettes:
#            silh = cv2.imread(silh_dir + sample_id + ".png", cv2.IMREAD_GRAYSCALE)
#            X_silh.append(silh)

        X_data = np.load(load_dir + sample_id + ".npz")

        X_params.append(X_data['smpl_params'])
        X_pcs.append(X_data['pointcloud'])
        X_silh.append(X_data['silhouette'])

    X_params = np.array(X_params)
    X_pcs = np.array(X_pcs)
    X_silh = np.array(X_silh)

    print("Finished.")
    if load_silhouettes:
        return X_params, X_pcs, X_silh
    else:
        return X_params, X_pcs


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


def format_offsetable_params(offsetable_params):
    param_ids = ["param_{:02d}".format(i) for i in range(85)]

    if offsetable_params == "all_pose":
        not_trainable = [0, 1, 2]
        #not_trainable = []
        offsetable_params_indices = [index for index in range(85) if index not in not_trainable and index < 72]
        #offsetable_params_indices = [index for index in range(85) if index not in not_trainable and index < 66]
        offsetable_params = [param_ids[index] for index in offsetable_params_indices]
    elif offsetable_params == "all_pose_and_global_rotation":
        not_trainable = [0, 2]
        #not_trainable = []
        offsetable_params_indices = [index for index in range(85) if index not in not_trainable and index < 72]
        #offsetable_params_indices = [index for index in range(85) if index not in not_trainable and index < 66]
        offsetable_params = [param_ids[index] for index in offsetable_params_indices]

    assert np.all([param in param_ids for param in offsetable_params])

    return offsetable_params


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
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31)), np.zeros((data_samples, 85))]
    elif ARCHITECTURE == "GroupedConv1DOptLearnerArchitecture":
        Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 31)), np.zeros((data_samples, 85))]
    else:
        raise ValueError("Architecture '{}' not recognised".format(ARCHITECTURE))

    return Y_data


def gather_input_data(data_samples, smpl, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, param_trainable, num_test_samples=5, MODE="RODRIGUES", LOAD_DATA_DIR=None):
    # Prepare initial input data
    zero_params = np.zeros(shape=(85,))
    zero_pc = smpl.set_params(beta=zero_params[72:82], pose=zero_params[0:72].reshape((24,3)), trans=zero_params[82:85])

    X_indices = np.array([i for i in range(data_samples)])
    zero_params = np.array([zero_params for i in range(data_samples)], dtype="float32")

    if LOAD_DATA_DIR is not None:
        # Load data from existing directory
        X_params, X_pcs = load_data(LOAD_DATA_DIR, num_samples=data_samples, load_silhouettes=False)
    else:
        # Generate the data
        if not all(value == 0.0 for value in POSE_OFFSET.values()):
            print("Offsetting parameters...")
            all_params = offset_params(zero_params, PARAMS_TO_OFFSET, POSE_OFFSET)
            if num_test_samples > 0:
                assert data_samples % num_test_samples == 0
                X_params = all_params[:num_test_samples]
            print("X_params shape: " + str(X_params.shape))
            print("Rendering parameters...")
            X_pcs = np.array([np.array(smpl.set_params(beta=params[72:82], pose=params[0:72].reshape((24, 3)), trans=params[82:85]).copy()) for params in X_params])
            print("X_pcs shape: " + str(X_pcs.shape))

            if num_test_samples > 0:
                times_to_repeat = int(data_samples/num_test_samples)

                X_pcs = X_pcs.copy()
                X_pcs = np.tile(X_pcs, (times_to_repeat, 1, 1))
                print("X_pcs shape: " + str(X_pcs.shape))

                X_params = np.tile(X_params, (times_to_repeat, 1))
                #X_params = np.array([X_params for i in range(data_samples/num_test_samples)], dtype=np.float32).reshape((data_samples, 85))
                print("X_params shape: " + str(X_params.shape))
        else:
            zero_params = np.zeros(shape=(85,))
            zero_pc = smpl.set_params(beta=zero_params[72:82], pose=zero_params[0:72].reshape((24,3)), trans=zero_params[82:85])
            #print("zero_pc: " + str(zero_pc))
            X_params = zero_params
            X_pcs = np.array([zero_pc for i in range(data_samples)], dtype="float32")

    if MODE == "EULER":
        # Convert from Rodrigues to Euler angles
        X_params = rodrigues_to_euler(X_params, smpl)

    X_data = [np.array(X_indices), np.array(X_params), np.array(X_pcs)]
    Y_data = architecture_output_array(ARCHITECTURE, data_samples)

    if ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture":
        trainable_params_mask = [int(param_trainable[key]) for key in sorted(param_trainable.keys(), key=lambda x: int(x[6:8]))]
        #print(trainable_params_mask)
        trainable_params_mask = np.tile(trainable_params_mask, (data_samples, 1))
        print("trainable_maks_shape: " + str(trainable_params_mask.shape))
        X_data += [trainable_params_mask]

    return X_data, Y_data


def gather_cb_data(X_data, Y_data, data_samples, num_cb_samples=5, where="spread"):
    """ Gather data for callbacks """
    if where == "spread":
        cb_samples = np.linspace(0, data_samples, num_cb_samples, dtype=int)
        cb_samples[-1] -= 1
    elif where == "front":
        cb_samples = [i for i in range(num_cb_samples)]
    elif where == "back":
        cb_samples = [i for i in range(data_samples - num_cb_samples, data_samples)]
    print("samples for visualisation callback: " + str(cb_samples))
    X_cb = [entry[cb_samples] for entry in X_data]
    Y_cb = [entry[cb_samples] for entry in Y_data]
    cb_pcs = X_cb[2]
    silh_cb = []
    for pc in cb_pcs:
        silh = Mesh(pointcloud=pc).render_silhouette(show=False)
        silh_cb.append(silh)

    return X_cb, Y_cb, silh_cb


def format_joint_levels(joint_levels):
    if joint_levels is None:
        # Flat kinematic structure
        return [[i for i in range(85)], ]

    kinematic_levels = []
    for level in joint_levels:
        level_list = []
        for joint in level:
            j1 = 3*joint
            j2 = 3*joint + 1
            j3 = 3*joint + 2

            named_j1 = "param_{:02d}".format(j1)
            named_j2 = "param_{:02d}".format(j2)
            named_j3 = "param_{:02d}".format(j3)
            level_list += [named_j1, named_j2, named_j3]
        kinematic_levels.append(level_list)

    return kinematic_levels


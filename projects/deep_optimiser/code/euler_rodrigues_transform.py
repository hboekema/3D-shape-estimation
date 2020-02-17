import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
from posenet_maths_v5 import rotation_matrix_to_euler_angles
from smpl_np_rot_v6 import eulerAnglesToRotationMatrix


def rodrigues_to_euler(X_params, smpl):
    """ Convert from Rodrigues to Euler angles """
    data_samples = X_params.shape[0]
    pose_rodrigues = np.array([X_params[:, i:i+3] for i in range(0, 72, 3)])
    #print(pose_rodrigues[0][0])
    pose_rodrigues = pose_rodrigues.reshape((24, data_samples, 1, 3))
    #print(pose_rodrigues[0][0])
    print("pose_rodrigues shape: " + str(pose_rodrigues.shape))
    R = np.array([smpl.rodrigues(vector) for vector in pose_rodrigues])
    #print(R[0][0])
    #R = R.reshape((data_samples, 24, 3, 3))
    pose_params = np.array([[rotation_matrix_to_euler_angles(rot_mat) for rot_mat in param_rot_mats] for param_rot_mats in R])
    pose_params = pose_params.reshape((data_samples, 72))
    print("pose_params shape: " + str(pose_params.shape))
    print("other params shape: " + str(X_params[:, 72:85].shape))
    X_params = np.concatenate([pose_params, X_params[:, 72:85]], axis=1)
    print("X_params shape: " + str(X_params.shape))

    return X_params


def euler_to_rodrigues(X_params):
    """ Convert Euler angles to Rodrigues format """
    data_samples = X_params.shape[0]
    pose_euler = np.array([X_params[:, i:i+3] for i in range(0, 72, 3)])
    #print(pose_euler[0][0])
    #pose_euler = pose_euler.reshape((24, data_samples, 1, 3))
    #print(pose_euler[0][0])
    print("pose_euler shape: " + str(pose_euler.shape))
    #R = np.array([[eulerAnglesToRotationMatrix(vector) for vector in vectors] for vectors in pose_euler])
    #print("R shape: " + str(R.shape))
    #print(R[0][0])
    #R = R.reshape((data_samples, 24, 3, 3))

    #pose_params = np.array([[Rot.from_dcm(rot_mat).as_rotvec() for rot_mat in param_rot_mats] for param_rot_mats in R])
    pose_params = np.array([Rot.from_euler('xyz', vectors, degrees=False).as_rotvec() for vectors in pose_euler])
    print("pose_params shape: " + str(pose_params.shape))
    pose_params = pose_params.reshape((data_samples, 72))
    print("pose_params shape: " + str(pose_params.shape))
    print("other params shape: " + str(X_params[:, 72:85].shape))
    X_params = np.concatenate([pose_params, X_params[:, 72:85]], axis=1)
    print("X_params shape: " + str(X_params.shape))

    return X_params


if __name__ == "__main__":
    from smpl_np import SMPLModel

    data_samples = 100
    X_params = 0.2 * 2*(np.random.rand(data_samples, 85) - 0.5)
    smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    euler_params = rodrigues_to_euler(X_params, smpl)
    rodrigues_params = euler_to_rodrigues(euler_params)

    print("euler_params: " + str(euler_params[0]))
    print("X_params: " + str(X_params[0]))
    print("rodrigues_params: " + str(rodrigues_params[0]))
    print("Parameters that differ: " + str(np.isclose(X_params, rodrigues_params, rtol=1e-3)[0]))

    assert np.allclose(X_params, rodrigues_params, rtol=1e-3)



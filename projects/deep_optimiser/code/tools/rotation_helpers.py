import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.append('/data/cvfs/hjhb2/projects/deep_optimiser/code/')
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


def rodrigues_to_rot3D(rodrigues):
    r = Rot.from_rotvec(rodrigues)
    return r.as_dcm()


def angle_between_rot3D(R1, R2):
    # Batched angle between two SO3 matrices
    mat_product = np.matmul(R1, np.transpose(R2, (0, 2, 1)))
    #print("mat_product shape: " + str(mat_product.shape))
    matrix_trace = 0.5*(np.trace(mat_product, axis1=1, axis2=2) - 1)
    matrix_trace = np.clip(matrix_trace, -1., 1.)
    assert np.max(np.abs(matrix_trace)) <= 1., "matrix trace value should be within -1 to 1: " + str(matrix_trace)
    assert np.sum(np.isnan(matrix_trace)) == 0, "NaNs in matrix_trace array: " + str(matrix_trace)
    #angle = np.arccos(0.5*(np.trace(mat_product, axis1=1, axis2=2) - 1))
    angle = np.arccos(matrix_trace)
    assert (np.max(angle) <= np.pi and np.min(angle) >= 0), "angle should in range [0, pi]: max: " + str(np.max(angle)) + ", min:" + str(np.min(angle))
    assert np.sum(np.isnan(angle)) == 0, "NaNs in angle array: " + str(angle)
    return angle


def geodesic_error(rodrigues1, rodrigues2):
    rotvec1 = np.reshape(rodrigues1[:, :72], (-1, 24, 3))
    rotvec2 = np.reshape(rodrigues2[:, :72], (-1, 24, 3))

    joints_angles = np.zeros((rotvec1.shape[0], 24))
    for i in range(24):
        R1 = rodrigues_to_rot3D(rotvec1[:, i])
        R2 = rodrigues_to_rot3D(rotvec2[:, i])

        joints_angles[:, i] = angle_between_rot3D(R1, R2)

    return joints_angles


def batched_dot_product(v1, v2):
    return np.sum(v1*v2, axis=-1)


if __name__ == "__main__":
    from smpl_np import SMPLModel

    data_samples = 100

    random_gt = np.random.rand(data_samples, 3)
    print("random_gt shape: " + str(random_gt.shape))
    gt_norm = np.linalg.norm(random_gt, axis=-1, keepdims=True)
    print("gt_norm shape: " + str(gt_norm.shape))
    unit_gt = random_gt / gt_norm
    print("unit_gt shape: " + str(unit_gt.shape))
    same = unit_gt
    random_vec = np.random.rand(data_samples, 3)
    print("random_vec shape: " + str(random_vec.shape))
    vec_norm = np.linalg.norm(random_vec, axis=-1, keepdims=True)
    print("vec_norm shape: " + str(vec_norm.shape))
    unit_vec = random_vec / vec_norm
    print("unit_vec shape: " + str(unit_vec.shape))

    # Calculate the angle by projecting vectors onto unit sphere
    gt_rot3d = rodrigues_to_rot3D(unit_gt)
    print("gt_rot3d shape: " + str(gt_rot3d.shape))
    pred_rot3d = rodrigues_to_rot3D(same)
    print("pred_rot3d shape: " + str(pred_rot3d.shape))

    geodesic_angles = angle_between_rot3D(gt_rot3d, pred_rot3d)
    print("geodesic_angles shape: " + str(geodesic_angles.shape))

    tol = 1e-3
    assert np.all(geodesic_angles <= tol), "angles: " + str(geodesic_angles)
    exit(1)

    X_params = 0.2 * 2*(np.random.rand(data_samples, 85) - 0.5)
    smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    euler_params = rodrigues_to_euler(X_params, smpl)
    rodrigues_params = euler_to_rodrigues(euler_params)

    print("euler_params: " + str(euler_params[0]))
    print("X_params: " + str(X_params[0]))
    print("rodrigues_params: " + str(rodrigues_params[0]))
    print("Parameters that differ: " + str(np.isclose(X_params, rodrigues_params, rtol=1e-3)[0]))

    assert np.allclose(X_params, rodrigues_params, rtol=1e-3)



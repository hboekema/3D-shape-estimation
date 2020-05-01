
import keras.backend as K
from keras.layers import Lambda, Reshape, Concatenate


# Projection helpers
def rot3d_from_ortho6d(ortho6d, gram_schmidt=True):
    a1 = Lambda(lambda x: x[:, :, :, 0])(ortho6d)
    a2 = Lambda(lambda x: x[:, :, :, 1])(ortho6d)
    print("a1 shape: " + str(a1.shape))
    print("a2 shape: " + str(a2.shape))

    b1 = Lambda(lambda x: K.tf.nn.l2_normalize(x, dim=-1))(a1)
    print("b1 shape: " + str(b1.shape))
    if gram_schmidt:
        # Use Gram-Schmidt process to find orthogonal vectors
        b1_projected = Lambda(lambda x: K.tf.reduce_sum(K.tf.multiply(x[1], x[0]), keep_dims=True) * x[1])([a2, b1])
        b2 = Lambda(lambda x: K.tf.nn.l2_normalize(x[0] - x[1], dim=-1))([a2, b1_projected])
        print("b2 shape: " + str(b2.shape))
        b3 = Lambda(lambda x: K.tf.cross(x[0], x[1]))([b1, b2])
        print("b3 shape: " + str(b3.shape))
    else:
        # Use cross products to find orthogonal vectors in R^3
        b3 = Lambda(lambda x: K.tf.nn.l2_normalize(K.tf.cross(x[0], x[1]), dim=-1))([b1, a2])
        print("b3 shape: " + str(b3.shape))
        b2 = Lambda(lambda x: K.tf.cross(x[0], x[1]))([b1, b3])
        print("b2 shape: " + str(b2.shape))

    R = Lambda(lambda x: K.stack(x, axis=-1))([b1, b2, b3])
    print("R shape: " + str(R.shape))
    #R = Reshape((-1, 3, 3))(R)
    #print("R shape: " + str(R.shape))
    #exit(1)

    return R

def ortho6d_from_rot3d(rot3d):
    ortho6d = Lambda(lambda x: x[:, :, :, 0:2])(rot3d)
    print("ortho_6d shape: " + str(ortho6d.shape))
    #exit(1)
    return ortho6d

def rot3d_from_rodrigues(rodrigues):
    # Convert axis-angle vector of shape [batch_dim, num_joints, 3] to 3D rotation matrix of shape [batch_dim, num_joints, 3, 3]
    # get the angle of rotation and unit vector
    epsilon = 1e-8
    theta = Lambda(lambda x: K.tf.norm(x, axis=-1, keep_dims=True) + epsilon)(rodrigues)
    print("theta shape: " + str(theta.shape))
    unit_r = Lambda(lambda x: x[0]/x[1])([rodrigues, theta])
    print("unit_r shape: " + str(unit_r.shape))

    # Get x-, y-, and z- components of unit axis vector
    r_x = Lambda(lambda x: K.expand_dims(x[:, :, 0]))(unit_r)
    r_y = Lambda(lambda x: K.expand_dims(x[:, :, 1]))(unit_r)
    r_z = Lambda(lambda x: K.expand_dims(x[:, :, 2]))(unit_r)
    print("r_x shape: " + str(r_x.shape))

    r_xx = Lambda(lambda x: x[0]*x[1])([r_x, r_x])
    r_yy = Lambda(lambda x: x[0]*x[1])([r_y, r_y])
    r_zz = Lambda(lambda x: x[0]*x[1])([r_z, r_z])
    r_xy = Lambda(lambda x: x[0]*x[1])([r_x, r_y])
    r_xz = Lambda(lambda x: x[0]*x[1])([r_x, r_z])
    r_yz = Lambda(lambda x: x[0]*x[1])([r_y, r_z])
    print("r_xx shape: " + str(r_xx.shape))

    c = Lambda(lambda x: K.cos(x))(theta)
    s = Lambda(lambda x: K.sin(x))(theta)
    C = Lambda(lambda x: 1-x)(c)

    r_xs = Lambda(lambda x: x[0]*x[1])([r_x, s])
    r_ys = Lambda(lambda x: x[0]*x[1])([r_y, s])
    r_zs = Lambda(lambda x: x[0]*x[1])([r_z, s])

    # Construct rotation matrix
    R11 = Lambda(lambda x: x[0]*x[1] + x[2])([r_xx, C, c])
    R12 = Lambda(lambda x: x[0]*x[1] - x[2])([r_xy, C, r_zs])
    R13 = Lambda(lambda x: x[0]*x[1] + x[2])([r_xz, C, r_ys])
    R21 = Lambda(lambda x: x[0]*x[1] + x[2])([r_xy, C, r_zs])
    R22 = Lambda(lambda x: x[0]*x[1] + x[2])([r_yy, C, c])
    R23 = Lambda(lambda x: x[0]*x[1] - x[2])([r_yz, C, r_xs])
    R31 = Lambda(lambda x: x[0]*x[1] - x[2])([r_xz, C, r_ys])
    R32 = Lambda(lambda x: x[0]*x[1] + x[2])([r_yz, C, r_xs])
    R33 = Lambda(lambda x: x[0]*x[1] + x[2])([r_zz, C, c])
    print("R11 shape: " + str(R11.shape))

    R = Concatenate()([R11, R12, R13, R21, R22, R23, R31, R32, R33])
    print("R shape: " + str(R.shape))
    R = Reshape((-1, 3, 3))(R)
    print("R shape: " + str(R.shape))

    return R

def rodrigues_from_rot3d(rot3d):
    # Convert 3D rotation matrix to (positive) axis-angle vector
    # Gather elements of the rotation matrix
    R11 = Lambda(lambda x: K.expand_dims(x[:, :, 0, 0]))(rot3d)
    R12 = Lambda(lambda x: K.expand_dims(x[:, :, 0, 1]))(rot3d)
    R13 = Lambda(lambda x: K.expand_dims(x[:, :, 0, 2]))(rot3d)
    R21 = Lambda(lambda x: K.expand_dims(x[:, :, 1, 0]))(rot3d)
    R22 = Lambda(lambda x: K.expand_dims(x[:, :, 1, 1]))(rot3d)
    R23 = Lambda(lambda x: K.expand_dims(x[:, :, 1, 2]))(rot3d)
    R31 = Lambda(lambda x: K.expand_dims(x[:, :, 2, 0]))(rot3d)
    R32 = Lambda(lambda x: K.expand_dims(x[:, :, 2, 1]))(rot3d)
    R33 = Lambda(lambda x: K.expand_dims(x[:, :, 2, 2]))(rot3d)
    print("R11 shape: " + str(R11.shape))

    # Calculate un-normalised angle vector
    r_x = Lambda(lambda x: x[0] - x[1])([R32, R23])
    r_y = Lambda(lambda x: x[0] - x[1])([R13, R31])
    r_z = Lambda(lambda x: x[0] - x[1])([R21, R12])
    print("r_x shape: " + str(r_x.shape))

    r_xyz = Concatenate()([r_x, r_y, r_z])
    print("r_xyz shape: " + str(r_xyz.shape))
    epsilon = 1e-8
    r_magnitude = Lambda(lambda x: K.tf.norm(x, axis=-1, keep_dims=True) + epsilon)(r_xyz)
    print("r_magnitude shape: " + str(r_magnitude.shape))

    # Calculate angle
    trace = Lambda(lambda x: x[0] + x[1] + x[2])([R11, R22, R33])
    theta = Lambda(lambda x: K.tf.atan2(x[0], x[1] - 1))([r_magnitude, trace])
    print("theta shape: " +str(theta.shape))

    # Form compressed vector and return this
    r_unit = Lambda(lambda x: x[0]/x[1])([r_xyz, r_magnitude])
    r = Lambda(lambda x: x[0]*x[1])([theta, r_unit])
    print("r_unit shape: " + str(r_unit.shape))
    print("r shape: " + str(r.shape))

    return r

def rot3d_from_euler(euler):
    # Convert Euler-Rodrigues angles to a 3D rotation matrix (with order of rotation X-Y-Z)
    euler_x = Lambda(lambda x: x[:, :, 0])(euler)
    euler_y = Lambda(lambda x: x[:, :, 1])(euler)
    euler_z = Lambda(lambda x: x[:, :, 2])(euler)

    cx = Lambda(lambda x: K.cos(x))(euler_x)
    sx = Lambda(lambda x: K.sin(x))(euler_x)
    cy = Lambda(lambda x: K.cos(x))(euler_y)
    sy = Lambda(lambda x: K.sin(x))(euler_y)
    cz = Lambda(lambda x: K.cos(x))(euler_z)
    sz = Lambda(lambda x: K.sin(x))(euler_z)

    R11 = Lambda(lambda x: x[0]*x[1])([cy, cz])
    R12 = Lambda(lambda x: x[0]*x[1]*x[2] - x[3]*x[4])([sx, sy, cz, cx, sz])
    R13 = Lambda(lambda x: x[0]*x[1]*x[2] + x[3]*x[4])([cx, sy, cz, sx, sz])
    R21 = Lambda(lambda x: x[0]*x[1])([cy, sz])
    R22 = Lambda(lambda x: x[0]*x[1]*x[2] + x[3]*x[4])([sx, sy, sz, cx, cz])
    R23 = Lambda(lambda x: x[0]*x[1]*x[2] - x[3]*x[4])([cx, sy, sz, sx, cz])
    R31 = Lambda(lambda x: -x)(sy)
    R32 = Lambda(lambda x: x[0]*x[1])([sx, cy])
    R33 = Lambda(lambda x: x[0]*x[1])([cx, cy])
    print("R11 shape: " + str(R11.shape))

    R = Lambda(lambda x: K.stack(x, axis=-1))([R11, R12, R13, R21, R22, R23, R31, R32, R33])
    print("R shape: " + str(R.shape))
    R = Reshape((-1, 3, 3))(R)
    print("R shape: " + str(R.shape))
    #exit(1)

    return R

def euler_from_rot3d(rot3d):
    # Rotation matrix in SO(3) to Euler-Rodrigues form. Note that many-to-one mappings exist, so all possible forms within [-pi, pi] are returned
    # Gather salient rotation matrix entries
    R11 = Lambda(lambda x: x[:, :, 0, 0])(rot3d)
    R12 = Lambda(lambda x: x[:, :, 0, 1])(rot3d)
    R13 = Lambda(lambda x: x[:, :, 0, 2])(rot3d)
    R21 = Lambda(lambda x: x[:, :, 1, 0])(rot3d)
    R31 = Lambda(lambda x: x[:, :, 2, 0])(rot3d)
    R32 = Lambda(lambda x: x[:, :, 2, 1])(rot3d)
    R33 = Lambda(lambda x: x[:, :, 2, 2])(rot3d)

    # Calculate y-axis rotation
    pi = K.constant(np.pi)
    def normal_case(R11, R21, R31, R32, R33):
        y1 = Lambda(lambda x: -K.tf.asin(x))(R31)
        y2 = Lambda(lambda x: pi - x)(y1)

        # Calculate cosine of possible y-axis rotations
        cy1 = Lambda(lambda x: K.cos(x))(y1)
        cy2 = Lambda(lambda x: K.cos(x))(y2)

        # Sign-correct the remaining rotation terms
        R11_cy1 = Lambda(lambda x: x[0]/x[1])([R11, cy1])
        R11_cy2 = Lambda(lambda x: x[0]/x[1])([R11, cy2])
        R21_cy1 = Lambda(lambda x: x[0]/x[1])([R21, cy1])
        R21_cy2 = Lambda(lambda x: x[0]/x[1])([R21, cy2])
        R32_cy1 = Lambda(lambda x: x[0]/x[1])([R32, cy1])
        R32_cy2 = Lambda(lambda x: x[0]/x[1])([R32, cy2])
        R33_cy1 = Lambda(lambda x: x[0]/x[1])([R33, cy1])
        R33_cy2 = Lambda(lambda x: x[0]/x[1])([R33, cy2])

        # Find possible x- and z-axis rotations
        x1 = Lambda(lambda x: K.tf.atan2(x[0], x[1]))([R32_cy1, R33_cy1])
        x2 = Lambda(lambda x: K.tf.atan2(x[0], x[1]))([R32_cy2, R33_cy2])
        z1 = Lambda(lambda x: K.tf.atan2(x[0], x[1]))([R21_cy1, R11_cy1])
        z2 = Lambda(lambda x: K.tf.atan2(x[0], x[1]))([R21_cy2, R11_cy2])
        print("x1 shape: " + str(x1.shape))

        # Gather corresponding elements
        #euler1 = Concatenate()([x1, y1, z1])
        euler1 = Lambda(lambda x: K.stack(x, axis=-1))([x1, y1, z1])
        print("euler1 shape: " + str(euler1.shape))
        #euler1 = Reshape((-1, 3))(euler1)
        #print("euler1 shape: " + str(euler1.shape))
        #euler2 = Concatenate()([x2, y2, z2])
        euler2 = Lambda(lambda x: K.stack(x, axis=-1))([x2, y2, z2])
        print("euler2 shape: " + str(euler2.shape))
        #euler2 = Reshape((-1, 3))(euler2)
        #print("euler2 shape: " + str(euler2.shape))
        euler = Lambda(lambda x: K.stack(x, axis=-1))([euler1, euler2])
        print("euler shape: " + str(euler.shape))
        #euler = Reshape((-1, 3, 2))(euler)
        #print("euler shape: " + str(euler.shape))
        #exit(1)
        return euler

    euler = normal_case(R11, R21, R31, R32, R33)
    return euler


def delta_d_to_ortho6d(delta_d, reordered_indices=None):
    # Cast GT difference Rodrigues representation to 6D for loss calculation
    delta_d_pose = Lambda(lambda x: x[:, 0:72])(delta_d)
    print("delta_d_pose shape: " + str(delta_d_pose.shape))

    # Re-order indices
    if reordered_indices is None:
        reordered_indices = [i for i in range(72)]
    else:
        print("Re-ordering delta_d")
    assert len(reordered_indices) == 72
    delta_d_pose = Lambda(lambda x: K.tf.gather(x, reordered_indices, axis=-1))(delta_d_pose)
    print("delta_d_pose shape: " + str(delta_d_pose.shape))

    delta_d_pose_vec = Reshape((24, 3))(delta_d_pose)
    print("delta_d_pose_vec shape: " + str(delta_d_pose_vec.shape))

    rot3d_delta_d_pose = rot3d_from_rodrigues(delta_d_pose_vec)
    print("rot3d_delta_d_pose shape: " + str(rot3d_delta_d_pose.shape))

    mapped_delta_d_pose_vec = ortho6d_from_rot3d(rot3d_delta_d_pose)
    print("mapped_delta_d_pose_vec shape: " + str(mapped_delta_d_pose_vec.shape))

    mapped_delta_d_pose = Reshape((24*3*2,))(mapped_delta_d_pose_vec)
    print("mapped_delta_d_pose shape: " + str(mapped_delta_d_pose.shape))

    return mapped_delta_d_pose


def ortho6d_pose_to_rodrigues(ortho6d_pose):
    # Reshape mapped predictions into vectors
    mapped_pose_vec = Reshape((24, 3, 2))(ortho6d)
    print("mapped_pose shape: " + str(mapped_pose_vec.shape))

    # Convert 6D representation to rotation matrix in SO(3)
    rot3d_pose = rot3d_from_ortho6d(mapped_pose_vec)
    print("rot3d_pose shape: " + str(rot3d_pose.shape))

    # Apply rotation in Rodrigues angles
    rodrigues_pose = rodrigues_from_rot3d(rot3d_pose)
    print("rodrigues_pose shape: " + str(rodrigues_pose.shape))
    delta_d_hat_pose = Reshape((72,))(rodrigues_pose)
    print("delta_d_hat_pose shape: " + str(delta_d_hat_pose.shape))

    return delta_d_hat_pose


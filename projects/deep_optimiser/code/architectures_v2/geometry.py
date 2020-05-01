
import keras.backend as K
from keras.layers import Lambda, Flatten


def get_mesh_normals(pc, faces, layer_name="mesh_normals"):
    """ Gather sets of points and compute their cross product to get (normalised) mesh normals """
    p0 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,0]).astype(np.int32), axis=-2))(pc)
    p1 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,1]).astype(np.int32), axis=-2))(pc)
    p2 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,2]).astype(np.int32), axis=-2))(pc)
    print("p0 shape: " + str(p0.shape))
    print("p1 shape: " + str(p1.shape))
    print("p2 shape: " + str(p2.shape))
    vec1 = Lambda(lambda x: x[1] - x[0])([p0, p1])
    vec2 = Lambda(lambda x: x[1] - x[0])([p0, p2])
    print("vec1 shape: " + str(vec1.shape))
    print("vec2 shape: " + str(vec2.shape))
    normals = Lambda(lambda x: K.l2_normalize(K.tf.cross(x[0], x[1]), axis=-1), name=layer_name)([vec1, vec2])

    return normals

def get_vertices(extra_inputs=False):
    # In order of: right hand, right wrist, right forearm, right bicep end, right bicep, right shoulder, top of cranium, left shoulder, left bicep, left bicep end, left forearm, left wrist, left hand,
    # chest, belly/belly button, back of neck, upper back, central back, lower back/tailbone,
    # left foot, left over-ankle, left shin, left over-knee, left quadricep, left hip, right, hip, right, quadricep, right over-knee, right shin, right, over-ankle, right foot
    if not extra_inputs:
        vertex_list = [5674, 5705, 5039, 5151, 4977, 4198, 411, 606, 1506, 1682, 1571, 2244, 2212,
            3074, 3500, 460, 2878, 3014, 3021,
            3365, 4606, 4588, 4671, 6877, 1799, 5262, 3479, 1187, 1102, 1120, 6740]
    else:
        # with added vertices in the feet
        vertex_list = [5674, 5705, 5039, 5151, 4977, 4198, 411, 606, 1506, 1682, 1571, 2244, 2212,
            3074, 3500, 460, 2878, 3014, 3021,
            3365, 4606, 4588, 4671, 6877, 1799, 5262, 3479, 1187, 1102, 1120, 6740, 3392, 3545, 3438, 6838, 6781, 6792]
    #face_array = np.array([11396, 8620, 7866, 5431, 6460, 1732, 4507])

    return vertex_list

def get_normals_from_vertices(gt_pc, optlearner_pc, vertex_list):
    face_array = np.array([[face for face in faces if vertex in face][0] for vertex in vertex_list])      # only take a single face for each vertex
    print("face_array shape: " + str(face_array.shape))
    gt_normals = get_mesh_normals(gt_pc, face_array, layer_name="gt_cross_product")
    print("gt_normals shape: " + str(gt_normals.shape))
    opt_normals = get_mesh_normals(optlearner_pc, face_array, layer_name="opt_cross_product")
    print("opt_normals shape: " + str(opt_normals.shape))
    #exit(1)

    return gt_normals, opt_normals

def get_vertex_diff(gt_pc, optlearner_pc, vertex_list):
    pc_euclidean_diff_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_diff) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    vertex_diff_NOGRAD = Lambda(lambda x: K.tf.gather(x, np.array(vertex_list).astype(np.int32), axis=-2))(pc_euclidean_diff_NOGRAD)
    print("vertex_diff_NOGRAD shape: " + str(vertex_diff_NOGRAD.shape))
    vertex_diff_NOGRAD = Flatten()(vertex_diff_NOGRAD)
    #exit(1)

    return vertex_diff_NOGRAD

def get_normals_diff(gt_normals, opt_normals):
    diff_normals = Lambda(lambda x: K.tf.cross(x[0], x[1]), name="diff_cross_product")([gt_normals, opt_normals])

    return diff_normals

def get_normals_angle_diff(gt_normals, opt_normals):
    diff_angles = Lambda(lambda x: K.tf.subtract(x[0], x[1]), name="diff_angle")([gt_normals, opt_normals])
    #diff_angles_norm_NOGRAD = Lambda(lambda x: K.tf.norm(x, axis=-1), name="diff_angle_norm")(diff_angles_NOGRAD)
    print("diff_angles shape: " + str(diff_angles.shape))
    dist_angles = Lambda(lambda x: K.mean(K.square(x), axis=-1), name="diff_angle_mse")(diff_angles)
    print("dist_angles shape: " + str(dist_angles.shape))

    return diff_angles, dist_angles



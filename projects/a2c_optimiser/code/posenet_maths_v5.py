import numpy as np
import math

def normalize_quaternion(q):
	return q/np.linalg.norm(q)
def angle_between_quaternions(q0,q1): #in radians
	q0q1=quaternion_multiply(q0,quaternion_conjugate(q1))
	assert np.abs(np.linalg.norm(np.array(q0q1))-1)<0.00001
	nq0q1=normalize_quaternion(q0q1)
	return 2*np.arccos(np.abs(nq0q1[0]))

def quaternion_conjugate(q):
	return np.array([q[0],-q[1],-q[2],-q[3]],dtype=np.float64)
def quaternion_rotate(q0,q1):
	result=quaternion_multiply(q0,q1)
	assert np.abs(np.linalg.norm(result)-1)<0.00001, 'Non-normalized quaternion'
	return result / np.linalg.norm(result)
def quaternion_multiply(q0, q1):
	w0, x0, y0, z0 = q0[0],q0[1],q0[2],q0[3]
	w1, x1, y1, z1 = q1[0],q1[1],q1[2],q1[3]
	return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)



def rotate_points_by_matrix(points,R):
        assert assert_rotation_matrix(R), 'Invalid rotation matrix'
	assert len(points.shape) == 2 and points.shape[0]==3
	result=  np.matmul(R,points)
	assert result.shape ==points.shape
	return result
def rotate_points_by_quaternion(points,q):
        assert np.abs(np.linalg.norm(q)-1)<0.00001, 'Non-normalized quaternion'
        assert len(points.shape) == 2 and points.shape[0]==3, ' points shape = '+str(points.shape)
	R = quaternion_to_rotation_matrix(q)
	p_R = rotate_points_by_matrix(points,R)
	points_exp = np.zeros((4,points.shape[1]))
	points_exp[1:,:]=points[:,:]
	def mult1(q_inp):
		return quaternion_multiply(quaternion_conjugate(q),q_inp)
	pmq = np.apply_along_axis(mult1,0,points_exp)
	def mult2(q_inp):
		return quaternion_multiply(q_inp,q)
	p_Q=np.apply_along_axis(mult2,0,pmq)[1:,:]
	assert p_Q.shape == points.shape
	assert np.max(np.abs(p_R-p_Q))<0.00001
	return p_Q
def adjust_quaternion_to_equirectangular_shift(crop_center_w, image_width, q, rolled):
	assert q.dtype ==np.float64
	assert np.abs(np.linalg.norm(np.array(q))-1)<0.00001, 'Non-normalized quaternion'
	shift = crop_center_w-image_width/2.0
	if( rolled == True):
		shift = shift + image_width/2.0
	theta = (shift/(1.0*image_width)) * 2*np.pi
	horizontalQ=euler_to_quaternion([0, -theta, 0])
	horizontalR = quaternion_to_rotation_matrix(horizontalQ)
	R = quaternion_to_rotation_matrix(q)
	R_new = np.matmul(R.T, horizontalR).T
	q_new = rotation_matrix_to_quaternion(R_new)
        q_new_q = quaternion_conjugate(quaternion_rotate(horizontalQ,quaternion_conjugate(q)))
	assert np.abs(np.linalg.norm(q_new_q)-1)<0.00001, 'Non-normalized quaternion'
        assert (np.max(np.abs(q_new_q-q_new))<0.00001) or (np.max(np.abs(q_new_q+q_new))<0.00001), 'q_neq '+str(q_new)+ ' q_new_q '+str(q_new_q)
	angle = angle_between_quaternions(q,q_new)
	angle2 = angle_between_quaternions(q_new,q)
	angle3 = angle_between_quaternions(q,q_new_q)
	assert np.abs(np.abs(angle)-np.abs(angle2))<0.00001, 'angle = '+str(angle)+ ' angle2 = '+str(angle2)
	assert np.abs(np.abs(angle)-np.abs(angle3))<0.00001, 'angle = '+str(angle)+ ' angle2 = '+str(angle3)
	assert ((np.abs(np.abs(angle)-np.abs(theta))<0.00001) or (np.abs(np.abs(angle)-np.abs(theta-np.pi*2))<0.00001)), 'angle = '+str(angle)+ ' theta = ' +str(theta)
	assert np.abs(np.linalg.norm(q_new)-1)<0.00001, 'Non-normalized quaternion'
	return q_new

def adjust_quaternion_to_equirectangular_shift_quaternion_only(crop_center_w, image_width, q, rolled):
        assert np.abs(np.linalg.norm(np.array(q))-1)<0.00001, 'Non-normalized quaternion'
        shift = crop_center_w-image_width/2.0
        if( rolled == True):
                shift = shift + image_width/2.0
        theta = (shift/(1.0*image_width)) * 2*np.pi
        horizontalQ=euler_to_quaternion([0, -theta, 0])
        q_new = quaternion_conjugate(quaternion_rotate(horizontalQ,quaternion_conjugate(q)))
        assert np.abs(np.linalg.norm(q_new)-1)<0.00001, 'Non-normalized quaternion'
        angle = angle_between_quaternions(q,q_new)
        angle2 = angle_between_quaternions(q_new,q)
        assert np.abs(np.abs(angle)-np.abs(angle2))<0.00001, 'angle = '+str(angle)+ ' angle2 = '+str(angle2)
        assert ((np.abs(np.abs(angle)-np.abs(theta))<0.00001) or (np.abs(np.abs(angle)-np.abs(theta-np.pi*2))<0.00001)), 'angle = '+str(angle)+ ' theta = ' +str(theta)
        assert np.abs(np.linalg.norm(q_new)-1)<0.00001, 'Non-normalized quaternion'

	q_new_q_R = adjust_quaternion_to_equirectangular_shift(crop_center_w,image_width,q,rolled)  #can be disabled for speed
	assert (np.max(np.abs(q_new_q_R-q_new))<0.00001) or (np.max(np.abs(q_new_q_R+q_new))<0.00001)	    #can be disabled for speed
        return q_new_q

def obtain_opensfmRT_from_posenetQT(Q_posenet,T_posenet):
	assert np.abs(np.linalg.norm(Q_posenet)-1)<0.00001, 'Non-normalized quaternion'
	R_opensfm = quaternion_to_rotation_matrix(Q_posenet)
	T_opensfm = -rotate_points_by_quaternion(np.reshape(T_posenet,(3,1)),Q_posenet)[:,0]
	T_opensfm2 = -np.matmul(R_opensfm,T_posenet)
	assert np.max(np.abs(T_opensfm2-T_opensfm))<0.00001
	Q2, T2 = obtain_posenetQT_from_opensfmRT(R_opensfm,T_opensfm)
	assert np.max(np.abs(Q2-Q_posenet))<0.00001
	assert np.max(np.abs(T2-T_posenet))<0.00001
	return R_opensfm,T_opensfm

def obtain_posenetQT_from_opensfmRT(R_opensfm,T_opensfm):
	assert assert_rotation_matrix(R_opensfm), 'Invalid rotation matrix'

	T_posenet = -np.matmul(R_opensfm.T,T_opensfm)
	Q_posenet = rotation_matrix_to_quaternion(R_opensfm)
	assert np.abs(np.linalg.norm(Q_posenet)-1)<0.00001, 'Non-normalized quaternion'

	return Q_posenet,T_posenet


def adjust_opensfm_RT_with_new_q(q,R_opensfm,T_opensfm):
	assert np.abs(np.linalg.norm(q)-1)<0.0001, 'Non-normalized quaternion'
	assert assert_rotation_matrix(R_opensfm), 'Invalid rotation matrix'
	O_t = -np.matmul(R_opensfm.T,T_opensfm)
	R_new = quaternion_to_rotation_matrix(q)	
	t_new = -np.matmul(R_new,O_t)
	return R_new,t_new

def euler_to_quaternion(euc):
        roll =euc[0]
        pitch=euc[1]
        yaw=euc[2]
        cy = np.cos(yaw/2)
        sy = np.sin(yaw/2)
        cr = np.cos(roll/2)
        sr = np.sin(roll/2)
        cp = np.cos(pitch/2)
        sp = np.sin(pitch/2)

        w= cy * cr * cp + sy * sr * sp
        x = cy * sr * cp - sy * cr * sp
        y = cy * cr * sp + sy * sr * cp
        z = sy * cr * cp - cy * sr * sp

	result= np.array([w,x,y,z],dtype=np.float64)
	assert np.abs(np.linalg.norm(result)-1)<0.00001
	return result / np.linalg.norm(result)

def quaternion_to_rotation_matrix(q_f):
        #q_f=quaternion.as_float_array(q)
        r=q_f[0]
        i=q_f[1]
        j=q_f[2]
        k=q_f[3]
        s = 1.0/(r*r+i*i+j*j+k*k)
        R = np.array([[1-2*s*(j*j+k*k), 2*s*(i*j-k*r),          2*s*(i*k+j*r)],
                      [2*s*(i*j+k*r),   1-2*s*(i*i+k*k),        2*s*(j*k-i*r)],
                      [2*s*(i*k-j*r),   2*s*(j*k+i*r),          1-2*s*(i*i+j*j)]],dtype=np.float64)

        assert_rotation_matrix(R)
        return R

def assert_rotation_matrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
def rotation_matrix_to_euler_angles(R) :
 
    assert(assert_rotation_matrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z],dtype=np.float64)

def rotation_matrix_to_quaternion(R):
	euler=rotation_matrix_to_euler_angles(R)
	q=euler_to_quaternion(euler)
	return q

if __name__ == "__main__":
	points = np.array([[13,4,5],[1,4,0],[0,0,0],[1,43,4],[3,54,5]])
	print('points shape '+str(points.shape))
	image_width =480
	center_crop_y =image_width/2+96

	rolled = False
	for i in range(455):
		q=euler_to_quaternion(np.random.rand(3)*np.pi) 
		#q=np.array([ 0,  0,  0, 1],dtype=np.float64)
	
		p_q=rotate_points_by_quaternion(points.T,q)

		R,T= obtain_opensfmRT_from_posenetQT(q,np.array([4,5,4]))
		#print('R = ' +str(R)+ ' T = '+str(T))
		q_new= adjust_quaternion_to_equirectangular_shift(center_crop_y,image_width,q, rolled)
		#print('New quaternion: '+str(q_new))
		#print('Angle: '+str(angle_between_quaternions(q,q_new)))
		#print('Finished')


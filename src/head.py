import cv2
import numpy as np
import dlib
import time
import math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
POINTS_NUM_LANDMARK = 68

# 获取最大的人脸
def _largest_face(dets):
    if len(dets) == 1:
        return 0

    face_areas = [ (det.right()-det.left())*(det.bottom()-det.top()) for det in dets]

    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(dets)):
        if face_areas[index] > largest_area :
            largest_index = index
            largest_area = face_areas[index]

    print("largest_face index is {} in {} faces".format(largest_index, len(dets)))

    return largest_index

# 用dlib检测关键点，返回姿态估计需要的几个点坐标
def get_image_points(img):
                            
    #gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )  # 图片调整为灰色
    dets = detector( img, 0 )

    if 0 == len( dets ):
        print( "ERROR: found no face" )
        return -1, None
    largest_index = _largest_face(dets)
    face_rectangle = dets[largest_index]

    landmark_shape = predictor(img, face_rectangle)

    return get_image_points_from_landmark_shape(landmark_shape)

# 用dlib检测关键点，返回姿态估计需要的几个点坐标
def get_single_face_image_points(img):
                            
    #gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )  # 图片调整为灰色
    dets = detector( img, 0 )

    if 0 == len( dets ):
        print( "ERROR: found no face" )
        return -1, None
    face_rectangle = dets[0]

    landmark_shape = predictor(img, face_rectangle)

    return get_image_points_from_landmark_shape(landmark_shape)

# 从dlib的检测结果抽取姿态估计需要的点坐标
def get_image_points_from_landmark_shape(landmark_shape):
    if landmark_shape.num_parts != POINTS_NUM_LANDMARK:
        print("ERROR:landmark_shape.num_parts-{}".format(landmark_shape.num_parts))
        return -1, None
    
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                (landmark_shape.part(30).x, landmark_shape.part(30).y),     # Nose tip //
                                (landmark_shape.part(8).x, landmark_shape.part(8).y),       # Chin
                                (landmark_shape.part(36).x, landmark_shape.part(36).y),     # Left eye left corner
                                (landmark_shape.part(45).x, landmark_shape.part(45).y),     # Right eye right corne
                                (landmark_shape.part(48).x, landmark_shape.part(48).y),     # Left Mouth corner
                                (landmark_shape.part(54).x, landmark_shape.part(54).y)      # Right mouth corner
                            ], dtype="double")

    return 0, image_points

    # 获取旋转向量和平移向量                        
def get_pose_estimation(img_size, image_points ):
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                             
                            ])
     
    # Camera internals
     
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
     
    print("Camera Matrix :{}".format(camera_matrix))
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE )
 
    print("Rotation Vector:\n {}".format(rotation_vector))
    print("Translation Vector:\n {}".format(translation_vector))
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs

# 从旋转向量转换为欧拉角
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    
    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta
    
    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)
    
    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)
    
    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    
    print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
    
	# 单位转换：将弧度转换为度
    Y = int((pitch/math.pi)*180)
    X = int((yaw/math.pi)*180)
    Z = int((roll/math.pi)*180)
    
    return 0, Y, X, Z
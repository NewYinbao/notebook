'''
Author: 
Date: 2020-08-19 08:52:42
LastEditTime: 2020-08-23 16:16:36
LastEditors: Please set LastEditors
Description: 哨岗相机标定相关函数
FilePath: .\src\cores.py
'''

import cv2
import numpy as np
import glob


#################################################################
# Calculate camera distortion coefficients
#################################################################
def getCameraCalibrationCoefficients(chessboardname, nx, ny):
    '''
    description:  计算相机畸变参数
    param 
        chessboardname, list, images path
        nx, grid x of chessboard
        ny, grid y of chessboard
    return 
        [ret, mtx, dist, rvecs, tvecs] is 
        retval, cameraMatrix, distCoeffs, rvecs, tvecs
    '''
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob(chessboardname)
    if len(images) > 0:
        print("images num for calibration : ", len(images))
    else:
        print("No image for calibration.")
        return
    
    ret_count = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img = cv2.resize(img, (960, 540))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (img.shape[1], img.shape[0])
        # Finde the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        img = cv2.drawChessboardCorners(gray, (nx,ny), corners, ret)
        cv2.imshow('corner', gray)
        cv2.waitKey(50)
        # If found, add object points, image points
        if ret == True:
            ret_count += 1
            objpoints.append(objp)
            imgpoints.append(corners)
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print('Do calibration successfully')
    return ret, mtx, dist, rvecs, tvecs


#################################################################
# Undistort image
#################################################################
def undistortImage(distortImage, mtx, dist):
    '''
    description: 图像去畸变
    param 
        distortImage, mtx, dist
    return 
        校正后的图像
    '''
    return cv2.undistort(distortImage, mtx, dist, None, mtx)


def getCameraLocation(model_points, image_points, mtx, dist):
    '''
    description: 计算哨岗相机的位置
    param
        model_points 世界坐标xo yo zo
        image_points 图像坐标x y
        mtx, dist 相机内参
    return 
        rotM 相机旋转矩阵
        tvecs 相机平移矩阵
    '''
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if success:
        print("Rotation Vector:\n {0}".format(rotation_vector*180/np.pi))
        print("Translation Vector:\n {0}".format(translation_vector))
    
        rotM = cv2.Rodrigues(rotation_vector)[0]
        position = -np.matrix(rotM).T * np.matrix(translation_vector)
        print('rotation Matrix is \n {0}'.format(rotM))
        print('camera position is \n {0}'.format(position))

        return success, rotM, translation_vector
    else:
        print('solvePnP error!')
        return success, None, None

    
def Estimatelocation(x, y, zo, mtx, dist, rotM, tvecs):
    '''
    description: 估算位置，强制Zo已知
    param {type} 
    return 世界坐标系的三维坐标点
    '''
    mtxinv = np.linalg.inv(mtx)
    rotMinv = rotM.T
    scale = (zo + np.dot(rotMinv[2], tvecs)) / np.dot((np.dot(rotMinv[2], mtxinv)), [x,y,1])

    pc = np.dot(mtxinv, [x,y,1]) * scale
    pc = pc.reshape((3,1))
    po = np.dot(rotMinv, pc) - np.dot(rotMinv, tvecs)

    return po


# read Coefficients
def readCameraMtx(path='./config/Coefficients.txt'):
    mtx = []
    dist = []
    with open(path, 'r') as f:
        for _ in range(3):
            mtx.append(list(map(float, f.readline().split(" "))))
        dist.append(list(map(float, f.readline().split(" "))))
    mtx = np.array(mtx)
    dist = np.array(dist)
    print('read camera coeffients from file:' + path)
    return mtx, dist

# save Coefficients
def saveCameraMtx(mtx, dist, path='./config/Coefficients.txt'):
    with open(path, 'w') as f:
        for row in mtx:
            f.write(' '.join(map(str, row)))
            f.write('\n')
        f.write(' '.join(map(str, dist[0])))
        print('save camera coefficients to '+path)

# save Extrinsic params
def saveExtrinsicparam(rotM, tvecs, path='./config/Extrinsicparam.txt'):
    with  open(path, 'w') as f:
        for row in rotM:
            f.write(' '.join(map(str, row)))
            f.write('\n')
        f.write(' '.join(map(str, tvecs.reshape((3,)))))
        print('save Extrinsic params to '+path)

# read Extrinsic params
def readExtrinsicparam(path='./config/Extrinsicparam.txt'):
    rotM = []
    tvecs = []
    with open(path, 'r') as f:
        for _ in range(3):
            rotM.append(list(map(float, f.readline().split(" "))))
        tvecs.append(list(map(float, f.readline().split(" "))))
    rotM = np.array(rotM)
    tvecs = np.array(tvecs).reshape((3,1))
    print('read camera coeffients from file:' + path)
    return rotM, tvecs

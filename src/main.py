'''
Author: 
Date: 2020-08-01 08:52:22
LastEditTime: 2020-08-23 19:57:38
LastEditors: Please set LastEditors
Description: 测试哨岗相机标定
FilePath: .\src\main.py
'''


import argparse

from cores import *

parser = argparse.ArgumentParser(description="your script description")
parser.add_argument('--mode', '-m', default='cameraLocation', help='select mode: cameraCalibrate 标记相机内参; cameraLocation  标记相机外参') 
    

def onmouse(event, x, y, flags, param):   #标准鼠标交互函数
    if event==cv2.EVENT_LBUTTONDOWN:      #当鼠标移动时
        param[:] = [x, y] 
        # print(x,y)          #显示鼠标所在像素的数值，注意像素表示方法和坐标位置的不同


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.mode == 'cameraCalibrate':
        print('开始相机标定：')
        imagepaths = './asserts/chessboard/4/*0.jpg'
        ret, mtx, dist, rvecs, tvecs = getCameraCalibrationCoefficients(imagepaths, 9, 6)

        print(mtx, dist)
        # save Coefficients
        saveCameraMtx(mtx, dist)

        # read Coefficients
        mtx, dist = readCameraMtx()
        
        # Read distorted chessboard image
        test_distort_image = cv2.imread('./assrets/chessboard/4/simple_indoor1_5watcher40000.jpg')
        test_distort_image = cv2.resize(test_distort_image,(960,540))
        
        # Do undistortion
        test_undistort_image = undistortImage(test_distort_image, mtx, dist)

        cv2.imshow('original', test_distort_image)
        cv2.imshow('undistorted', test_undistort_image)
        cv2.waitKey(0)

    elif args.mode == 'cameraLocation':
        print('开始标定相机外参：')
        
        # 读取图片、预处理
        image = cv2.imread('./asserts/simple_indoor1_5watcher40005.jpg')
        image = cv2.resize(image, (960,540))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

        # 提取联通域，获得标记的亮点坐标
        _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (0,0,255), 3)
        imagepoints = []
        for i in range(len(contours)):
            p = contours[i]
            if len(p) < 10:
                continue
            x = np.mean(p[:,0,0:1])
            y = np.mean(p[:,0,1:2])
            imagepoints.append([x, y])
            image = cv2.circle(image, (int(x),int(y)), 3, (0, 0, 255), -1)
            
        # 图像坐标与世界坐标对应，很重要
        # index = list(map(int, input('please input image points orders: ').split(' ')))
        # index = [3,2,4,12,0,7,6,8,13,1,10,9,11,14,5]
        # for i in index:
        #     temp.append(imagepoints[i])
        
        # 图像坐标与世界坐标对应 改进版 鼠标点击最近的
        mouse_position = [0,0]
        last_position = [0,0]
        temp = []
        cv2.namedWindow("image")     
        cv2.setMouseCallback("image", onmouse, mouse_position)
        cv2.imshow('image', image)
        cv2.imshow("img",img) 
        i = 0
        while imagepoints:
            if last_position[0] != mouse_position[0] and last_position[1] != mouse_position[1]: 
                print(mouse_position)
                d = np.abs(np.array(mouse_position) - np.array(imagepoints))
                d = np.sum(d, axis=-1)
                index = np.argmin(d)
                
                i += 1
                x,y = imagepoints[index]
                image=cv2.putText(image,str(i),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,0),2)

                temp.append(imagepoints.pop(index))
                cv2.imshow('image', image)

            last_position[:] = mouse_position[:]
            if cv2.waitKey(20) == ord('q'):break   #按下'q'键，退出

        imagepoints = np.array(temp)

        # 读取世界坐标点
        realpoints = []
        with open('./asserts/realcoord.txt', 'r') as f:
            for i in range(15):
                realpoints.append(list(map(float, f.readline().split(' '))))
        realpoints = np.array(realpoints)
        
        # 读取相机内参
        mtx, dist = readCameraMtx()

        # 标定相机外参
        t = min(len(imagepoints), len(realpoints)) - 1 # 最后一个留作验证
        _, rotM, tvecs = getCameraLocation(realpoints[:t], imagepoints[:t], mtx, dist)
        
        # 保存外参矩阵
        saveExtrinsicparam(rotM, tvecs)

        # 读取外参矩阵
        rotM, tvecs = readExtrinsicparam()

        # 估计最后一个坐标点的的世界坐标，zo已知
        po = Estimatelocation(temp[-1][0], temp[-1][1], -87, mtx, dist, rotM, tvecs)
        print('估计的位置：\n', po.reshape((3,)))
        print('实际的位置：\n', realpoints[-1])
        cv2.destroyAllWindows()
        
    else:
        print('这个指令还没开发呢，现在有：')
        print('cameraCalibrate 标记相机内参；')
        print('cameraLocation  标记相机外参')
import cv2
import numpy as np
import os
from tqdm import tqdm


no_camera = False
if no_camera:
    data_path = './dataset/Calibration/new_camera/'
else:
    data_path = './dataset/Calibration/fr1_rgb_calibration/'


if no_camera:

    camera = cv2.VideoCapture(0)
    i = 0
    while 1:
        (grabbed, img) = camera.read()
        cv2.imshow('img',img)

        # press 'j' to save
        if cv2.waitKey(1) & 0xFF == ord('j'): 

            i += 1
            filename = data_path + str(i) + '.png'
            cv2.imwrite(filename, img)
            print('image saved:', filename)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



# 找棋盘格角点
# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值

#棋盘格模板规格
w = 8   # 10 - 1
h = 6   # 7  - 1

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w * h, 3), np.float32)
objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1,2)
objp = objp * 18.1  # 18.1 mm

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点


#加载图像
img_names = os.listdir(data_path)
img_names.sort()
images = []
for img_name in tqdm(img_names):
    img_path = data_path + img_name
    images.append(img_path)



i=0
for fname in images:

    img = cv2.imread(fname)

    #获取图像的长宽
    h1, w1 = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    u, v = img.shape[:2]

    # 找到棋盘格角点
    # ret -- bool, finding successfully or not
    # corners -- found corner points on chessboard
    ret, corners = cv2.findChessboardCorners(image = gray, patternSize = (w,h), flags= None)

    # 如果找到足够点对，将其存储起来
    if ret == True:

        print(f"find corners in image {fname}:")
        i = i+1

        # 在原角点的基础上寻找亚像素角点
        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        #追加进入世界三维点和平面二维点中
        objpoints.append(objp)
        imgpoints.append(corners)

        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 640, 480)
        cv2.imshow('findCorners',img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

print('--------------- calculating -------------------')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


print("ret:",ret)
print("mtx:\n",mtx)      # 内参数矩阵
print("dist畸变值:\n",dist   )   # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs旋转(向量)外参:\n",rvecs)   # 旋转向量  # 外参数
print("tvecs平移(向量)外参:\n",tvecs  )  # 平移向量  # 外参数
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
print('newcameramtx外参', newcameramtx)


no_camera = False
if no_camera:

    #打开摄像机
    camera=cv2.VideoCapture(0)

    while True:

        (grabbed,frame)=camera.read()
        h1, w1 = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))

        # 纠正畸变
        dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        mapx,mapy=cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w1,h1),5)
        dst2=cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)

        # 裁剪图像，输出纠正畸变以后的图片
        x, y, w1, h1 = roi
        dst1 = dst1[y:y + h1, x:x + w1]

        cv2.imshow('dst2', dst2)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q保存一张图片
            cv2.imwrite("../u4/frame.jpg", dst1)
            break

    camera.release()
    cv2.destroyAllWindows()



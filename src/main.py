import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def calibrate_camera(images_path):
    images = glob.glob(images_path)

    objpoints = []
    imgpoints = []

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, : 2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for fname in images:
        img = mpimg.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

            # img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)

    image0 = mpimg.imread(images[0])
    gray = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

def undistort(mtx, dist, frame_path):
    test_img_name = frame_path
    test_img = mpimg.imread(test_img_name)

    undist = cv2.undistort(test_img, mtx, dist, None, mtx)

    return undist

images = '../data/camera_cal/calibration*.jpg'
test_img = '../data/test_images/straight_lines1.jpg'
mtx, dist = calibrate_camera(images)
undist_img = undistort(mtx, dist, test_img)
plt.imshow(undist_img)
plt.show()

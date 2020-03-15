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


def threshold(img, s_thresh=(180, 255), sx_thresh=(50, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    # # Stack each channel
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return combined_binary

# test the pipeline here
images = '../data/camera_cal/calibration*.jpg'
test_img = '../data/test_images/straight_lines1.jpg'
undistorted_img = '../data/output_images/undostorted.jpg'
bin_img_name = '../data/output_images/thresholded.jpg'

mtx, dist = calibrate_camera(images)
undist_img = undistort(mtx, dist, test_img)
bin_img = threshold(undist_img)
# plt.imshow(undist_img)
# plt.show()
# undist_img = cv2.cvtColor(undist_img, cv2.COLOR_RGB2BGR)
# cv2.imwrite(undistorted_img, undist_img)
# cv2.imwrite(bin_img_name, bin_img * 255)
bin_half = bin_img[bin_img.shape[0] // 2:, :]
plt.imshow(bin_half)
plt.show()

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


def threshold(img, s_thresh=(180, 255), sx_thresh=(30, 100)):
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


def warp(bin_img):
    # src coordinates for perspective transform
    x1 = 190
    y1 = bin_img.shape[0]
    x2 = 590
    y2 = 450
    x3 = 690
    y3 = 450
    x4 = 1120
    y4 = bin_img.shape[0]
    src = np.float32(
        [[x1, y1],
         [x2, y2],
         [x3, y3],
         [x4, y4]])

    # dst coords
    x1_dst = 350
    y1_dst = bin_img.shape[0]
    x2_dst = 350
    y2_dst = 0
    x3_dst = 950
    y3_dst = 0
    x4_dst = 950
    y4_dst = bin_img.shape[0]
    dst = np.float32(
        [[x1_dst, y1_dst],
         [x2_dst, y2_dst],
         [x3_dst, y3_dst],
         [x4_dst, y4_dst]])

    img_size = (bin_img.shape[1], bin_img.shape[0])

    src_draw = np.array(
        [[x1, y1],
         [x2, y2],
         [x3, y3],
         [x4, y4]])
    dst_draw = np.array(
        [[x1_dst, y1_dst],
         [x2_dst, y2_dst],
         [x3_dst, y3_dst],
         [x4_dst, y4_dst]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bin_img, M, img_size, flags=cv2.INTER_LINEAR)
    # check if the coordinates are correct
    # color_img = np.dstack((bin_img, np.zeros_like(bin_img), np.zeros_like(bin_img))) * 255
    # color_img = cv2.polylines(color_img, [src_draw], True, (0, 255, 0))
    # cv2.imshow('src', color_img)
    # cv2.waitKey()
    return warped


# test the pipeline here
images = '../data/camera_cal/calibration*.jpg'
test_img_name = '../data/test_images/straight_lines1.jpg'
undistorted_img_name = '../data/output_images/undostorted.jpg'
bin_img_name = '../data/output_images/thresholded.jpg'
warped_img_name = '../data/output_images/warped.jpg'

mtx, dist = calibrate_camera(images)
undist_img = undistort(mtx, dist, test_img_name)
binary_img = threshold(undist_img)
warped = warp(binary_img)
# plt.imshow(undist_img)
# plt.show()
# undist_img = cv2.cvtColor(undist_img, cv2.COLOR_RGB2BGR)
# cv2.imwrite(undistorted_img, undist_img)
# cv2.imwrite(bin_img_name, binary_img * 255)
# cv2.imwrite(warped_img_name, warped * 255)

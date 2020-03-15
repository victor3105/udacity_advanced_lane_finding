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


def threshold(img, s_thresh=(160, 255), sx_thresh=(50, 100)):
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
    x1 = 180
    y1 = bin_img.shape[0]
    x2 = 590
    y2 = 450
    x3 = 690
    y3 = 450
    x4 = 1150
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
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(bin_img, M, img_size, flags=cv2.INTER_LINEAR)
    # check if the coordinates are correct
    # color_img = np.dstack((bin_img, np.zeros_like(bin_img), np.zeros_like(bin_img))) * 255
    # color_img = cv2.polylines(color_img, [src_draw], True, (0, 255, 0))
    # cv2.imshow('src', color_img)
    # cv2.waitKey()
    return warped, Minv


def find_lines(warped):
    # find line pixels from scratch
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((warped, warped, warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window + 1) * window_height
        win_y_high = warped.shape[0] - window * window_height
        win_xleft_low = np.int(leftx_current - margin)
        win_xleft_high = np.int(leftx_current + margin)
        win_xright_low = np.int(rightx_current - margin)
        win_xright_high = np.int(rightx_current + margin)

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if (len(good_left_inds) > minpix):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if (len(good_right_inds) > minpix):
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(warped, leftx, lefty, rightx, righty, find_lines_out_img):
    # Find our lane pixels first
    # leftx, lefty, rightx, righty, out_img = find_lines(warped)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # ## Visualization ##
    # # Colors in the left and right lane regions
    # find_lines_out_img[lefty, leftx] = [255, 0, 0]
    # find_lines_out_img[righty, rightx] = [0, 0, 255]
    #
    # plt.imshow(find_lines_out_img)
    # # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.show()



    return left_fit, right_fit, ploty, left_fitx, right_fitx


def search_around_poly(warped, left_fit, right_fit, find_lines_out_img):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_win_low = left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] - margin
    left_win_high = left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] + margin
    right_win_low = right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] - margin
    right_win_high = right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] + margin
    left_lane_inds = (nonzerox > left_win_low) & (nonzerox < left_win_high)
    right_lane_inds = (nonzerox > right_win_low) & (nonzerox < right_win_high)

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit, right_fit, ploty, left_fitx, right_fitx = fit_polynomial(warped, leftx, lefty, rightx, righty, find_lines_out_img)

    return left_fit, right_fit, ploty, left_fitx, right_fitx


def calc_world_parameters(img, left_fit, right_fit, ploty):
    y_scale = 30 / 720
    x_scale = 3.7 / 700

    # y coordinate where the curvature radius will be calculated
    y = np.max(ploty)

    left_curverad = (1 + (2 * left_fit[0] * y * y_scale + left_fit[1]) ** 2) ** (3 / 2) \
                    / np.abs(2 * left_fit[0])
    right_curverad = (1 + (2 * right_fit[0] * y * y_scale + right_fit[1]) ** 2) ** (3 / 2)\
                     / np.abs(2 * right_fit[0])

    x_left = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    x_right = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
    lane_center = np.mean([x_left, x_right])
    img_center = img.shape[1] // 2
    distance_info = 'Vehicle is '
    distance = round((img_center - lane_center) * x_scale * 100) / 100
    if distance > 0:
        distance_info += str(distance) + ' m right of center'
    else:
        distance_info += str(abs(distance)) + ' m left of center'


    curve_rad = np.mean([left_curverad, right_curverad])

    # use 2 decimal digits
    curve_rad = round(curve_rad * 100) / 100

    radius_info = 'Radius of curvature: ' + str(curve_rad) + ' m'
    font = cv2.FONT_HERSHEY_PLAIN
    color = (255, 255, 0)
    corner1 = (10, 30)
    corner2 = (10, 60)
    thickness = 1
    font_scale = 2
    img = cv2.putText(img, radius_info, corner1, font, font_scale, color, thickness, cv2.LINE_AA)
    img = cv2.putText(img, distance_info, corner2, font, font_scale, color, thickness, cv2.LINE_AA)
    plt.imshow(img)
    plt.show()


def draw_lane_plane(img, left_fitx, right_fitx, Minv):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result
    # plt.imshow(result)
    # plt.show()

# test the pipeline here
images = '../data/camera_cal/calibration*.jpg'
test_img_name = '../data/test_images/straight_lines2.jpg'
undistorted_img_name = '../data/output_images/undostorted.jpg'
bin_img_name = '../data/output_images/thresholded.jpg'
warped_img_name = '../data/output_images/warped.jpg'

mtx, dist = calibrate_camera(images)
undist_img = undistort(mtx, dist, test_img_name)
binary_img = threshold(undist_img)
warped, Minv = warp(binary_img)
leftx, lefty, rightx, righty, out_img = find_lines(warped)
left_fit, right_fit, ploty, left_fitx, right_fitx = fit_polynomial(warped, leftx, lefty, rightx, righty, out_img)
img_w_plane = draw_lane_plane(undist_img, left_fitx, right_fitx, Minv)
calc_world_parameters(img_w_plane, left_fit, right_fit, ploty)

# plt.imshow(img)
# plt.show()

# fig, axs = plt.subplots(2)
# axs[0].imshow(warped)
# axs[1].plot(histogram)
# plt.show()
# plt.imshow(undist_img)
# plt.show()
undist_img = cv2.cvtColor(undist_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(undistorted_img_name, undist_img)
cv2.imwrite(bin_img_name, binary_img * 255)
cv2.imwrite(warped_img_name, warped * 255)

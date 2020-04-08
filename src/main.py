import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def calibrate_camera(images_path):
    '''
    Perform camera calibration
    :param images_path: path to chessboard images
    :return: mtx: camera matrix; dist: distortion coefficients
    '''
    images = glob.glob(images_path)

    objpoints = []
    imgpoints = []

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, : 2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for fname in images:
        img = mpimg.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

            # img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)

    image0 = mpimg.imread(images[0])
    gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist


def undistort(mtx, dist, img):
    '''
    Perform current frame undistortion
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param img: frame to be distorted
    :return: undist - undistorted image
    '''
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist


# 160 255 50 100
def threshold(img, s_thresh=(160, 255), sx_thresh=(40, 100)):
    '''
    Perform image binarization using saturation and Sobel edges
    :param img: image to binarize
    :param s_thresh: saturation range for binarization
    :param sx_thresh: gradient value range for binarization
    :return: combined_binary: binarized image
    '''
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
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
    '''
    Transform image to bird-eye view
    :param bin_img: binary image to be processed
    :return: warped: transformed image; Minv: matrix for inverse perspectiveTransformation (for visualization)
    '''
    src = np.float32([[490, 482], [810, 482],
                      [1250, 720], [40, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                      [1250, 720], [40, 720]])
    img_size = (bin_img.shape[1], bin_img.shape[0])

    src_draw = np.array([[490, 482], [810, 482],
                        [1250, 720], [40, 720]])
    dst_draw = np.array([[0, 0], [1280, 0],
                        [1250, 720], [40, 720]])

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
    '''
    Find lane lines using sliding windows method
    :param warped: transformed bird-eye image
    :return: leftx: x coordinates of left line pixels
    lefty: y coordinates of left line pixels
    rightx: x coordinates of right line pixels
    righty: y coordinates of right line pixels
    out_img: image for visualization of lines search
    '''
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
    '''
    Fit curves to the detected lines
    :param warped: warped image
    :param leftx: x coordinates of left line pixels
    :param lefty: y coordinates of left line pixels
    :param rightx: x coordinates of right line pixels
    :param righty: y coordinates of right line pixels
    :param find_lines_out_img: image from the previous step (for visualization)
    :return:
    left_fit: polynomial coefficients for the left line
    right_fit: polynomial coefficients for the right line
    ploty: y coordinates for calculation of x values
    left_fitx: calculated x coordinates of the left line
    right_fitx: calculated x coordinates of the right line
    '''
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


def search_around_poly(warped, left_fit, right_fit, find_lines_out_img, left_line, right_line):
    '''
    Fit lines around the previously found lines
    :param warped: warped (bird-eye) image
    :param left_fit: polynomial coefficients for the left line
    :param right_fit: polynomial coefficients for the right line
    :param find_lines_out_img: image for visualization
    :param left_line: instance of Line class for parameters averaging and smoothing over iterations
    :param right_line: instance of Line class for parameters averaging and smoothing over iterations
    :return:
    left_fit_m: average of polynomial coefficients
    right_fit_m: average of polynomial coefficients
    ploty: set of y values (from top to bottom)
    left_fitx: calculated x values for the left line
    right_fitx: calculated x values for the right line
    '''
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
    if len(left_line.best_fit) == 15:
        left_line.best_fit.pop()
        right_line.best_fit.pop()
    left_line.best_fit.append(left_fit)
    right_line.best_fit.append(right_fit)
    left_fit_m = np.mean(left_line.best_fit, axis=0)
    right_fit_m = np.mean(right_line.best_fit, axis=0)

    return left_fit_m, right_fit_m, ploty, left_fitx, right_fitx


def calc_world_parameters(img, left_fit, right_fit, ploty, left_line, right_line):
    '''
    Calculate radius of curvature of the lines and distance to the lane center
    and show this information in output image
    :param img:
    :param left_fit:
    :param right_fit:
    :param ploty:
    :param left_line:
    :param right_line:
    :return:
    '''
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

    # calculate the lane center x-coordinate
    lane_center = np.mean([x_left, x_right])
    # we suppose that camera is installed in the center of the vehicle
    img_center = img.shape[1] // 2
    distance_info = 'Vehicle is '
    # thus, the distance from the car center to the lane center is:
    distance = round((img_center - lane_center) * x_scale * 100) / 100
    if distance > 0:
        distance_info += str(distance) + ' m right of center'
    else:
        distance_info += str(abs(distance)) + ' m left of center'
    # fine average values over several iterations (frames)
    if len(left_line.radius_of_curvature) == 40:
        left_line.radius_of_curvature.pop()
        right_line.radius_of_curvature.pop()
    left_line.radius_of_curvature.append(left_curverad)
    right_line.radius_of_curvature.append(right_curverad)
    left_rad = np.mean(left_line.radius_of_curvature)
    right_rad = np.mean(right_line.radius_of_curvature)
    curve_rad = np.mean([left_rad, right_rad])

    # use 2 decimal digits
    curve_rad = round(curve_rad * 100) / 100

    radius_info = 'Radius of curvature: ' + str(curve_rad) + ' m'
    font = cv2.FONT_HERSHEY_PLAIN
    color = (255, 255, 0)
    corner1 = (10, 30)
    corner2 = (10, 60)
    thickness = 2
    font_scale = 2
    img = cv2.putText(img, radius_info, corner1, font, font_scale, color, thickness, cv2.LINE_AA)
    img = cv2.putText(img, distance_info, corner2, font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def draw_lane_plane(img, left_fitx, right_fitx, ploty, Minv):
    '''
    Draw lane plane in output image in green
    :param img: image to draw in
    :param left_fitx: x coordinates of the left line
    :param right_fitx: x coordinates of the right line
    :param ploty: y values
    :param Minv: inverse transform camera matrix
    :return:
    result: the output image
    '''
    warp_zero = np.zeros_like(img[:, :, 0]).astype(np.uint8)
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


def process_frame(frame, frame_number, prev_left_fit, prev_right_fit, mtx, dist, left_line, right_line):
    '''
    Perform all the processing steps on current frame
    :param frame: frame to process
    :param frame_number: the number of the current frame
    :param prev_left_fit: previous value of the polynomial coefficients
    :param prev_right_fit: previous value of the polynomial coefficients
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param left_line: Lane class instance to keep left line parameters
    :param right_line: Lane class instance to keep right line parameters
    :return:
    left_fit: averaged polynomial coefficients for the left line
    right_fit: averaged polynomial coefficients for the right line
    res: output image
    '''
    undist_img = undistort(mtx, dist, frame)
    binary_img = threshold(undist_img)
    warped, Minv = warp(binary_img)
    # plt.imshow(warped)
    # plt.show()
    out_img = np.zeros_like(frame)  # placeholder
    if frame_number == 0:
        leftx, lefty, rightx, righty, out_img = find_lines(warped)
        left_fit, right_fit, ploty, left_fitx, right_fitx = fit_polynomial(warped, leftx, lefty, rightx, righty, out_img)
    else:
        left_fit, right_fit, ploty, left_fitx, right_fitx = search_around_poly(warped, prev_left_fit, prev_right_fit, out_img, left_line, right_line)
    img_w_plane = draw_lane_plane(undist_img, left_fitx, right_fitx, ploty, Minv)
    res = calc_world_parameters(img_w_plane, left_fit, right_fit, ploty, left_line, right_line)
    # cv2.imshow('res', res)
    # cv2.waitKey()

    return left_fit, right_fit, res


class Line():
    def __init__(self):
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = []

# #test the pipeline here
images = '../data/camera_cal/calibration*.jpg'
test_img_name = '../data/test_images/test1.jpg'
calib_undist_name = '../data/output_images/calib_undist.jpg'
undistorted_img_name = '../data/output_images/undostorted.jpg'
bin_img_name = '../data/output_images/thresholded.jpg'
warped_img_name = '../data/output_images/warped.jpg'

test_img = mpimg.imread(test_img_name)

prev_l_fit = np.array([0, 0, 0])
prev_r_fit = np.array([0, 0, 0])
# calibrate camera
mtx, dist = calibrate_camera('../data/camera_cal/calibration*.jpg')
# calib_0 = cv2.imread('../data/camera_cal/calibration3.jpg')
# undist_calib_img = undistort(mtx, dist, calib_0)
# cv2.imwrite(calib_undist_name, undist_calib_img)
input_video_name = '../data/project_video.mp4'
vidcap = cv2.VideoCapture(input_video_name)
output_video_name = '../data/out_1.mp4'
# open the input video
success, frame = vidcap.read()
# create the output video
out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame.shape[1], frame.shape[0]))
count = 0
left_line = Line()
right_line = Line()
while success:
    # process input frames
    print('Processing frame ' + str(count))
    prev_l_fit, prev_r_fit, res = process_frame(frame, count, prev_l_fit, prev_r_fit, mtx, dist, left_line, right_line)
    # write the output frame
    out.write(res)
    success, frame = vidcap.read()
    count += 1

out.release()

# for i in range(1):
#     prev_l_fit, prev_r_fit, res = process_frame(test_img, i, prev_l_fit, prev_r_fit)
#     plt.imshow(res)
#     plt.show()

#
# mtx, dist = calibrate_camera(images)
# undist_img = undistort(mtx, dist, test_img)
# binary_img = threshold(undist_img)
# warped, Minv = warp(binary_img)
# leftx, lefty, rightx, righty, out_img = find_lines(warped)
# left_fit, right_fit, ploty, left_fitx, right_fitx = fit_polynomial(warped, leftx, lefty, rightx, righty, out_img)
# img_w_plane = draw_lane_plane(undist_img, left_fitx, right_fitx, ploty, Minv)
# calc_world_parameters(img_w_plane, left_fit, right_fit, ploty)

# plt.imshow(img)
# plt.show()

# fig, axs = plt.subplots(2)
# axs[0].imshow(warped)
# axs[1].plot(histogram)
# plt.show()
# plt.imshow(undist_img)
# plt.show()
# undist_img = cv2.cvtColor(undist_img, cv2.COLOR_RGB2BGR)
# cv2.imwrite(undistorted_img_name, undist_img)
# cv2.imwrite(bin_img_name, binary_img * 255)
# cv2.imwrite(warped_img_name, warped * 255)

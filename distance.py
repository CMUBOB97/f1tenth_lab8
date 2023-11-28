"""
This is the script to calibrate the camera, including:
- find camera intrinsic matrix and distortion
- find camera mounting height by the sample cone image

The cone reference coordinate is selected by clicking the image in imshow

Authors:
    Chi Gao (chig@andrew.cmu.edu)
    Chris Klammer (cklammer@andrew.cmu.edu)
    Nathan Litzinger (nlitzing@andrew.cmu.edu)
    Tianhao Ye (ty2@andrew.cmu.edu)

Last modified: 11/26/2023
"""
import cv2
import glob
import numpy as np

# debug flag for debug print
DEBUG = 0

# constants of chessboard corners
CHESS_WIDTH = 6
CHESS_HEIGHT = 8
# TODO: need to get the correct checker square size for this
#       the unit should be the same as cone distance below
CHESS_SQUARE_SIZE = 1

# known distance in X(car) of the cone image (unit: cm)
CONE_X_DIST = 40

# folder name for each dataset
calibration_path = "calibration/"
object_detection_path = "resource/"

# x, y coordinates of the reference point on the cone
# for camera mounting height info use
cone_img_x = 0
cone_img_y = 0

# define corner tuning criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# camera calibration procedure using chessboard corners
# return intrinsic matrix, distortion, rotation and translation vectors
def camera_intrinsic_calibration(folder_name):
    # get a list of all chessboard images
    img_files = glob.glob(folder_name + '*.png')
    img_counter = 1

    # prepare two arrays to store corner object points (3D) and image points (2D)
    objpoints = []
    imgpoints = []

    # define object points first (differ by x, y, on the same plane by z)
    objp = np.zeros((CHESS_WIDTH * CHESS_HEIGHT, 3), np.float32)
    objp[:, :2] = CHESS_SQUARE_SIZE * np.mgrid[0:CHESS_HEIGHT, 0:CHESS_WIDTH].T.reshape(-1, 2)
    
    # define window specs for corner detection
    if DEBUG == 1:
        cv2.namedWindow('corner result', cv2.WINDOW_AUTOSIZE)

    # loop through all images to append corners
    for img_file in img_files:
        
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find corners
        ret, corners = cv2.findChessboardCorners(gray, (CHESS_HEIGHT, CHESS_WIDTH), None)

        # if found, add points
        if ret == True:
            objpoints.append(objp)
            
            corners_subpixel = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
            imgpoints.append(corners_subpixel)
            
            # if debug flag is on, draw corners
            if DEBUG == 1:
                print("showing corner detection of image", img_counter, "out of", len(img_files), "images")
                img_counter += 1
                cv2.drawChessboardCorners(img, (CHESS_HEIGHT, CHESS_WIDTH), corners_subpixel, ret)
                cv2.imshow('corner result', img)
                cv2.waitKey(0)
                
    # use these points to calibrate camera intrinsics
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # check calibration results
    if DEBUG == 1:
        print("check reprojection error:")
        print(ret)
        print("check intrinsic matrix:")
        print(mtx)
        print("check distortion coefficients:")
        print(dist)
        print("check rotational vectors:")
        print(rvecs)
        print("check translational vectors:")
        print(tvecs)
        
    return mtx, dist, rvecs, tvecs

# undistort an image based on input distortion coefficients
def undistort_image(mtx, dist, img_file):
    # read image file
    img = cv2.imread(img_file)
    
    # undistort
    normal_img = cv2.undistort(img, mtx, dist, None, None)
    
    if DEBUG == 1:
        cv2.namedWindow('undistorted image', cv2.WINDOW_AUTOSIZE)
        print("check undistorted image:")
        cv2.imshow('undistorted image', normal_img)
        cv2.waitKey(0)
        
    return normal_img

# function that gets mouse click in an opened image
def get_mouse_click(event, x, y, flags, param):
    # set cone_img_x and y to be global
    global cone_img_x, cone_img_y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        cone_img_x = x
        cone_img_y = y
        print(f"point clicked at (x, y): ({x}, {y})")
        clicked_img = np.copy(cone_img)
        clicked_img = cv2.circle(clicked_img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('cone image', clicked_img)
        
# function that shows the cone image for corner selection
def select_cone_corner(cone_img_file):
    # make image global for callback function
    global cone_img
    
    # read image file
    cone_img = cv2.imread(cone_img_file)
    
    cv2.namedWindow('cone image', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('cone image', get_mouse_click)
    
    cv2.imshow('cone image', cone_img)
    cv2.waitKey(0)
    
# function that finds camera mounting height
def find_camera_mounting_height(mtx):
    # set cone_img_x and y to be global
    global cone_img_x, cone_img_y
    
    # acquire principal point y coordinate and y focal length from the camera matrix
    y_center = mtx[1, 2]
    y_focal = mtx[1, 1]
    
    # assuming that x, y coordinates of the cone is the known one
    # and it is on the ground, calculate the mounting height
    mounting_height = (cone_img_y - y_center) / y_focal * CONE_X_DIST
    print("camera mounting height:")
    print(mounting_height, "cm")
    
    return mounting_height

# function that guesses cone distance from the car
def find_cone_dist(mtx, mounting_height):
    # set cone_img_x and y to be global
    global cone_img_x, cone_img_y
    
    # acquire principal point y coordinate and y focal length from the camera matrix
    x_center = mtx[0, 2]
    y_center = mtx[1, 2]
    x_focal = mtx[0, 0]
    y_focal = mtx[1, 1]
    
    # calculate the distance accounting for the mounting height
    x_car_dist = (y_focal * mounting_height) / (cone_img_y - y_center)
    y_car_dist = (x_center - cone_img_x) * x_car_dist / x_focal
    print("estimated cone distance:")
    print(x_car_dist, "cm")
    print("estimated cone shift:")
    print(y_car_dist, "cm")
    print(f"the cone corner at car frame is at: ({x_car_dist}, {y_car_dist})")
    
    return x_car_dist

if __name__ == "__main__":
    
    # camera calibration first
    mtx, dist, rvecs, tvecs = camera_intrinsic_calibration(calibration_path)
    
    # get the cone image with known distance
    cone_known = object_detection_path + "cone_x40cm.png"
    
    # OPTIONAL: undistort the cone image
    # tried, but the result is not good, severe distortion on the edge of the image
    # cone_img_undistorted = undistort_image(mtx, dist, cone_known)
    
    # get reference cone corner by a click into the image
    select_cone_corner(cone_known)
    
    # find the camera mounting height
    cam_height = find_camera_mounting_height(mtx)
    
    # get the cone image without distance info
    cone_unknown = object_detection_path + "cone_unknown.png"
    
    # get reference cone corner by a click into the image
    select_cone_corner(cone_unknown)
    
    # now make the inference of the distance of cone in the unknown image
    cone_dist = find_cone_dist(mtx, cam_height)
    
    # destroy all windows
    cv2.destroyAllWindows()
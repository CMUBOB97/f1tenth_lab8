"""
This is the script to draw boundaries for lane segments

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

# hsv upper and lower bounds for the lane
# NOTE: normal HSV ranges are:
# H: 0 - 360 S: 0 - 100 V: 0 - 100
# but opencv uses
# H: 0 - 179 S: 0 - 255 V: 0 - 255
# so the values below are scaled
lane_lower_bound = np.array([20, 40, 160])
lane_upper_bound = np.array([50, 160, 255])


"""
This function contains the following steps:
- blur the original image
- mask the image according to target HSV
- create a segmented image (black and white)
- draw coutours around and overlay that on the original image
"""
def segment_lane_image(lane_img_file):
    
    # read the image
    lane_img = cv2.imread(lane_img_file)
    if DEBUG == 1:
        print("check original image:")
        cv2.namedWindow('original image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('original image', lane_img)
        cv2.waitKey(0)
    
    # smooth the image (blur)
    blur_img = cv2.blur(lane_img, (7, 7))
    if DEBUG == 1:
        print("check blurred image:")
        cv2.namedWindow('blurred image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('blurred image', blur_img)
        cv2.waitKey(0)
    
    blur_hsv = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
    
    # create a mask using the defined lower and upper bounds
    lane_mask = cv2.inRange(blur_hsv, lane_lower_bound, lane_upper_bound)
    
    # segment the image
    lane_filtered = cv2.bitwise_and(blur_img, blur_img, mask=lane_mask)
    lane_grayscale = cv2.cvtColor(lane_filtered, cv2.COLOR_BGR2GRAY)
    ret, lane_segmented = cv2.threshold(lane_grayscale, 127, 255, 0)
    if DEBUG == 1:
        print("check segmented image:")
        cv2.namedWindow('segmented image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('segmented image', lane_segmented)
        cv2.waitKey(0)
        
    # draw edges
    contours, hierarchy = cv2.findContours(lane_segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lane_img_drawn = cv2.drawContours(lane_img, contours, -1, (0, 255, 0), 3)
    print("check final result:")
    cv2.namedWindow('final result', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('final result', lane_img_drawn)
    cv2.waitKey(0)
        
    return lane_img_drawn

if __name__ == "__main__":
    
    # get the lane image path
    lane_img_file = "resource/lane.png"
    
    # segment it
    segment_lane_image(lane_img_file)
    
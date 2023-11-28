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
DEBUG = 1

# hsv upper and lower bounds for the lane
# NOTE: normal HSV ranges are:
# H: 0 - 360 S: 0 - 100 V: 0 - 100
# but opencv uses
# H: 0 - 179 S: 0 - 255 V: 0 - 255
# so the values below are scaled
lane_lower_bound = np.array([20, 40, 160])
lane_upper_bound = np.array([50, 160, 205])

"""
NOTE:
this is a very rough prototype that does not work yet.
we need to blur the image, set proper threshold, make
connectivity patches, do edge detections, etc.

feel free to add the remaining functions
"""
# segment the lane image based on HSV color scheme
def segment_lane_image(lane_img_file):
    
    # read the image
    lane_img = cv2.imread(lane_img_file)
    if DEBUG == 1:
        print("check original image:")
        cv2.namedWindow('original image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('original image', lane_img)
        cv2.waitKey(0)
    
    # smooth the image (blur)
    blur_img = cv2.blur(lane_img, (5, 5))
    if DEBUG == 1:
        cv2.namedWindow('blurred image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('blurred image', blur_img)
        cv2.waitKey(0)
    
    blur_hsv = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
    
    # create a mask using the defined lower and upper bounds
    lane_mask = cv2.inRange(blur_hsv, lane_lower_bound, lane_upper_bound)
    
    # segment the image
    lane_segmented = cv2.bitwise_and(lane_img, lane_img, mask=lane_mask)
    if DEBUG == 1:
        cv2.namedWindow('segmented image', cv2.WINDOW_AUTOSIZE)
        print("check segmented image:")
        cv2.imshow('segmented image', lane_segmented)
        cv2.waitKey(0)
    
    return lane_segmented

if __name__ == "__main__":
    
    # get the lane image path
    lane_img_file = "resource/lane.png"
    
    # segment it
    segment_lane_image(lane_img_file)
    
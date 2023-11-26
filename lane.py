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
# NOTE: this is not the right HSV range yet
lane_lower_bound = np.array([0, 75, 30])
lane_upper_bound = np.array([10, 100, 50])

"""
NOTE:
this is a very rough prototype that does not work yet.
we need to blur the image, set proper threshold, make
connectivity patches, do edge detections, etc.

feel free to add the remaining functions
"""
# segment the lane image based on HSV color scheme
def segment_lane_image(lane_img_file):
    
    # read the image in HSV color scheme
    lane_img = cv2.imread(lane_img_file)
    lane_hsv = cv2.cvtColor(lane_img, cv2.COLOR_BGR2HSV)
    
    # create a mask using the defined lower and upper bounds
    lane_mask = cv2.inRange(lane_hsv, lane_lower_bound, lane_upper_bound)
    
    # segment the image
    lane_segmented = cv2.bitwise_and(lane_img, lane_img, mask=lane_mask)
    if DEBUG == 1:
        cv2.namedWindow('original image', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('segmented image', cv2.WINDOW_AUTOSIZE)
        print("check original image:")
        cv2.imshow('original image', lane_img)
        cv2.waitKey(0)
        print("check segmented image:")
        cv2.imshow('segmented image', lane_segmented)
        cv2.waitKey(0)
    
    return lane_segmented

if __name__ == "__main__":
    
    # get the lane image path
    lane_img_file = "resource/lane.png"
    
    # segment it
    segment_lane_image(lane_img_file)
    
from scipy.spatial.distance import cdist 
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import winsound


def hough_circles_acc(img, r):
    # create new H
    H = np.zeros(img.shape)
    Htemp = np.zeros(img.shape)

    # get coordinates of the image and map to lin index 
    # go through all the edge points, flip x/y bc Q4
    # the hasmap index is 
    for (y, x), is_edge in np.ndenumerate(img):
        if is_edge:
            # draw a circle on in Htmep and add its vote to H
            temp = Htemp.copy()
            cv2.circle(temp, (x, y), r, 1, thickness=1)
            H += temp

    # all circles are drawn onto H, 
    # the places where all these circles overlap
    # will be the location of the circle as the votes from surrounding 
    # points will distinguish it
    return H
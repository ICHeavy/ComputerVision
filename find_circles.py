import numpy as np
import cv2
import matplotlib.pyplot as plt
from hough_peaks import hough_peaks
from hough_circles_acc import hough_circles_acc

def find_circles(img, radRng):
    # create 2D arr of coordinates 
    centers = np.empty((0, 2), int)
    radii = []
    # grab each lin index of coordinate via hashmap
    # i.e. [20,50] - list [20,50), indexed [0,29)
    for i, rad in enumerate(np.arange(radRng[0], radRng[1])):
        # travel through the range of radii
        oH = hough_circles_acc(img, rad)
        print('--------------------------- Radius: ', rad, ' -----------------------------')
        centerPeaks, oHtmep = hough_peaks(oH, 5)
        # grab centers off each radius, add those to both lists as you go
        centers = np.concatenate((centers, centerPeaks), axis=0)
        radlist = [rad]
        radii += radlist*len(centerPeaks)

    return centers, np.array(radii)
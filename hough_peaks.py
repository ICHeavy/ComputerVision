import numpy as np
import cv2
import matplotlib.pyplot as plt

def hough_peaks(H, Npeaks, border = 5):
    peaks = []
    Htemp = np.copy(H)
    # loop through number of peaks
    for i in range(Npeaks):
        # find max indices in full arr then get (x,y) location
        # get mem location of maxes, first
        # then, get coordinates in H
        # add coordinates to index arr
        # these are indexes of 
        idx = np.argmax(Htemp) 
        Hidx = np.unravel_index(idx, Htemp.shape)
        peaks.append(Hidx)

        # check if too close to the edges of the image
        # get x, y
        iy, ix = Hidx # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        v = border/2
        if (ix - v) < 0: 
            xmin = 0
        else: 
            xmin = ix - v
            xmin = int(round(xmin))

        if ((ix + v + 1) > H.shape[1]): 
            xmax = H.shape[1]
        else: 
            xmax = ix + v + 1
            xmax = int(round(xmax))

        # if idx_y is too close to the edges choose appropriate values
        if (iy - v) < 0: 
            ymin = 0
        else: 
            ymin = iy - v
            ymin = int(round(ymin))

        if ((iy + v+ 1) > H.shape[0]): 
            ymax = H.shape[0]
        else: 
            ymax = iy + v + 1
            ymax = int(round(ymax))

        # bound each index by the neighborhood size and set all values to 0
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                # remove neighborhoods in H1
                Htemp[y, x] = 0

                # highlight peaks in original H
                if (x == xmin or x == (xmax - 1)):
                    H[y, x] = 255
                if (y == ymin or y == (ymax - 1)):
                    H[y, x] = 255
    
    
# return the indicies and the original Hough space with selected points
    return peaks, H
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from hough_peaks import hough_peaks

def hough_lines_draw(img, peaks, theta_arr, rho_arr):
    numi = len(peaks)

    for i in range(numi):
        # reminder H[rho][theta] the peaks having same format 
        r = rho_arr[peaks[i][0]]
        t = theta_arr[peaks[i][1]]

        # polar -> cartesian
        a = np.cos(t)
        b = np.sin(t) 
        ar = a*r
        br = b*r

        # get start and end points, scale
        x1 = int(ar + 1000*(-b))
        y1 = int(br + 1000*(a))
        x2 = int(ar - 1000*(-b))
        y2 = int(br - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

import numpy as np
from math import *

def disparity_ssd(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    # TODO: Your code here
    # grab dims, set up window size and create disparity array
    r, c = L.shape

    window = (3,3)
    h, w = window
    disp = np.zeros_like(L)
    #truncate
    strips = r//h
    ranger = r - h
    for y in range(ranger):
        Lstrip = L[y : y+h, :]
        Rstrip = R[y : y+h, :]
        dispstrip = findStripDisp(Lstrip, Rstrip, w)
        disp[y, :] = dispstrip
    copyguy = np.copy(disp)
    return copyguy 

def findStripDisp(Lstrip, Rstrip, w):
    Lcol = Lstrip.shape[1]
    numBlocks = Lcol//w
    disp = np.zeros((Lcol,))

    for i in range(numBlocks):
        x = i * w
        templ = Lstrip[:, x : x+w]
        idealx = templSSD(templ, Rstrip) 
        disp[x : x+w] = x - idealx
    return disp


def templSSD(templ, Rstrip):
    idealx = 0
    th, tw = templ.shape
    numBlocks = Rstrip.shape[1]//tw
    minguy = np.inf

    for i in range(numBlocks):
        x = i * tw
        temptempl = Rstrip[:, x : x+tw]
        diff = np.sum((templ - temptempl)**2)
        if diff < minguy:
            minguy = diff
            idealx = x
    
    return idealx
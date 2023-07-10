from cv2 import TM_CCORR_NORMED
import numpy as np
import cv2
import matplotlib.pyplot as plt

def disparity_ncorr(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    # TODO: Your code here
    r, c = L.shape
    # block size
    tr = tc = 10
    rangeguy = 100
    disp = np.zeros_like(L)

    for x in range(tr / 2, r-tr / 2):
        trmin = max(x - tr/2, 0)
        trmax = min(x+tr/2 +1, r)

        for y in range (tc / 2, c-tc / 2):
            tcmin = max(y - tc/2, 0)
            tcmax = min(y+tc/2 +1, c)
            templ = L[trmin:trmax, tcmin:tcmax].astype(np.int32) 

            rightmin = max(y-rangeguy/2, 0)
            rightmax = min(y+rangeguy/2 +1, c)
            Rstrip = R[trmin:trmax, rightmin:rightmax]

            diff = cv2.matchTemplate(Rstrip, templ, method= TM_CCORR_NORMED)
            minguy = max(y-rightmin-tc /2, 0)

            dist = np.arange(diff.shape[1]) - minguy
            cost = diff - np.abs(dist)
            q,q,q, maxguy = cv2.minMaxLoc(cost)
            disp[x,y] = dist[maxguy[0]]
    
    return disp
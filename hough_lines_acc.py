from scipy.spatial.distance import cdist 
import numpy as np
import time
import matplotlib.pyplot as plt
import winsound


from PIL import Image, ImageFilter


def hough_lines_acc(img): 
    st = time.time()

    h,w = img.shape 
    D = int(np.ceil(np.sqrt(w * w + h * h)))

    # generate theta [-90:90]
    # THETA IN 180 DEG RANGE      
    theta_arr = np.deg2rad(np.arange(-90.0, 90.0))
    rho_arr = np.linspace(-D, D, D * 2)
    numThetas = len(theta_arr)

    # MAKE ARRAYS OF SIN AND COS INDEXES
    thetaCos = np.cos(theta_arr)
    thetaSin =np.sin(theta_arr)
    
    # acc array of doubles (H) has:
    # X axis -  theta [-90:90]
    # Y axis - rho (dist from origin)
    H = np.zeros((2*D, numThetas ), np.uint64)
    r, c = np.nonzero(img)  # (row, col) indexes of edges, but flipped
    
    #  for traveling down each y value
    # grab index of each nonzero value (of each edge)
    for i in range(len(c)):
        x = c[i]
        y = r[i]
        # go through each theta 
        # calculate the distance from that point to origin
        #  inc the accumulator arr
        print(i, "\n")
        for j in range(numThetas):
            rho = round(x*thetaCos[j] + y*thetaSin[j]) + D
            H[rho, j] += 1
        

    print("!!! FINALLY DONE !!!")
    winsound.Beep(800, 2000) # so i can go do other things

    return [H, theta_arr, rho_arr]
    
    #read in BW
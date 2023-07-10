# print("WELCOME ")
# print("READING IMAGE...")
# ############### READING IN PHOTO ##############################
# rawimg = cv2.imread(inPath + photo1, cv2.IMREAD_UNCHANGED)
# #test data
# X[m,n] 
    # m = # pixels
    # n = dimensions of color
# K = # clusters
# iters = # of iterations
# numpix,colors = (36,3)
# K = 3
# iters = 3
# 6x6 pixel photo, 3 color density
# set each pixel itensity value to a random value 
# make our input matrix
# X = [random.sample(range(0,255),3) for b in range(numpix)]

# 
# def kmeans_single(X, K, iters):
#     # returns ids[m] = list 1-K
#     # means[k,n] = list of means around each cluster
#     # ssd = sum of sqared ditances between points and assigned means over all clusters
#     # ^ yikes
#     numpix = len(X)
#     reds = []
#     greens = []
#     blues = []

#     if (K == 3):
            
#         for i in range(numpix):
#             #isolate colors for sanity reasons
#             reds.append(X[i][0])
#             greens.append(X[i][1])
#             blues.append(X[i][2])
        
#         # generate cluster centers
#             # generating a random int to be the center based on the min/max 
#             # of the respective color K(num of clusters) times, in this case 3
#         redCenters = [random.randint(min(reds), max(reds)) for b in range(K)]
#         greenCenters = [random.randint(min(greens), max(greens)) for c in range(K)]
#         blueCenters = [random.randint(min(blues), max(blues)) for d in range(K)]


#         print("X  : ")
#         print(X)
#         print(" REDS : ")
#         print(reds)
#         print(" GREENS : ")
#         print(greens)
#         print(" BLUES : ")
#         print(blues)
#         print(" RED CENTERS : ")
#         print(redCenters)
#         print(" BLUE CENTERS  : ")
#         print(blueCenters)
#         print(" GREEN CENTERS : ")
#         print(greenCenters)


#     elif(K == 1):
#         # todo
#         pass
#     else:
#         print("ERROR: K OUT OF SCOPE!")
#         return -1
#LIBRARIES
import cv2
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import random

def makeData(pts = 100, rad = 1, min = 0, max = 1, xcenter = 0, ycenter = 0):
    pi = np.pi
    r = rad * np.sqrt(np.random.uniform(min, max, size = pts)) 
    theta = np.random.uniform(min, max, size= pts)*2*pi

    x = xcenter + np.cos(theta) * r
    y = ycenter + np.sin(theta) * r

    x = np.round(x,3)
    y = np.round(y,3)

    df = np.column_stack([x,y])
    df = pd.DataFrame(df)
    df.columns = ['x','y']
    return(df)

def init_centers(X, k):
    #grab min/max and dims 
    n_dims = X.shape[1]
    cmin = X.min().min()
    cmax = X.max().max()
    centers = []

    # generate k number of rando nums for each dim
    for centers in range(k):
        center = np.random.uniform(cmin, cmax, n_dims)
        centers= np.append(centers, center)

    centers = pd.DataFrame(centers, columns= X.columns)
    return centers

def calcDist(a,b):
  # calculating the error is the same as calculating distance 
  # these formulas are the same but in this case 
  # are not performing the same type of distance
  # ergo distance from center value between X and centers
  # dist=(x2–x1)2+(y2–y1)2
    dist = np.square(np.sum((a-b)**2))
    return dist 


    

inPath = r'./input/'
outPath = "C:\\Users\\simaj\\Documents\\School\\CV\\ps1_python_Sima_John\\output"
photo1 = r"ps1-1-a-1.png"




reds = makeData(pts= 25, rad= 10, xcenter= 5,ycenter= 5)
greens = makeData(pts= 25, rad= 10, xcenter= 20,ycenter= 8)
blues = makeData(pts= 25, rad= 10, xcenter= 10,ycenter= 20)
X = reds.append(greens).append(blues)
X.head()
K = 10
distances = np.array([])
plt.scatter(reds['x'], reds['y'], c= 'r')
plt.scatter(greens['x'], greens['y'], c= 'g')
plt.scatter(blues['x'], blues['y'], c= 'b')
plt.show()
centers = init_centers(X,K)
for center in range(centers.shape[0]):
    dist = calcDist(centers.iloc[center, :2], X.iloc[0,:2])
    distances = np.append(distances, dist)

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

inPath = r'./input/'
outPath = r'./output/'

THREEDEENORMIE = r'pts3d-norm.txt'
ONETWODEENORMIE = r'pts2d-norm-pic_a.txt'
PIC_A = "pic_a.jpg"
PIC_A_2D = "pts2d-pic_a.txt"
PIC_A_2D_NORM = "pts2d-norm-pic_a.txt"
PIC_B = "pic_b.jpg"
PIC_B_2D = "pts2d-pic_b.txt"
SCENE = "pts3d.txt"


outphoto1 = r"ps4-1-a-1.png"
outphoto2a = r"ps4-2-a-1.png"

def getPts(filepath):
    pts = []
    with open(inPath+filepath) as f:
        for line in f:
            pts.append(tuple([float(i) for i in line.split()]))
    return pts

def getFmatrix(leftPts, rightPts):
    pts = []
    for (u, v, *blank), (utemp, vtemp, *blank) in zip(leftPts, rightPts):
        pts.append([utemp * u, utemp * v, utemp, vtemp * u, vtemp * v, vtemp, u, v, 1])
    return np.asarray(pts)

def SVDtrick(a):
    # find the eigenvector of ATA with smallest eigenvalue, that’s m
    u, v = np.linalg.eig(np.matmul(a.T, a))
    m = v[:, u.argmin()]
    return m

def getSVDm(pts_world, pts_proj):
    a = getAmatrix(pts_world, pts_proj)
    return SVDtrick(a).reshape((3, 4))

def calculate_avg_residual(m, pts_world, pts_proj):
    pts_world_mat = np.append(np.asarray(pts_world), np.ones((len(pts_world), 1)), axis=1).T
    pts_proj_est = np.matmul(m, pts_world_mat)
    pts_proj_est_non_homo = pts_proj_est.T[:, 0:2] / pts_proj_est.T[:, [2, 2]]
    residual = np.linalg.norm(pts_proj_est_non_homo - np.asarray(pts_proj))
    # return averaged residual
    return residual/np.sqrt(pts_world_mat.shape[0])

def getAmatrix(threeDee, twoDee):
    a = []
    for (x, y, z), (u, v) in zip(threeDee, twoDee):
        a.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        a.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    return np.asarray(a)

def leastSquares(twoDguy,threeDguy):
    # solve for the 3x4 matrix MnormA  given the normalized 2D and 3D lists
    # pts2d = (numpts, (u,v))
    # pts3d = (numpts, (x,y,z))
    A = getAmatrix(threeDguy, twoDguy)
    u, v = np.linalg.eig(np.matmul(A.T, A))
    return  v[:, u.argmin()].reshape((3, 4))


# 1.a
twoD = getPts(ONETWODEENORMIE)
threeD = getPts(THREEDEENORMIE)
# 
M = leastSquares(twoD,threeD)
print('The matrix M you recovered from the normalized points: \n', M)

threeDmatrix = np.append(np.asarray(threeD), np.ones((len(threeD), 1)), axis=1).T
# Test it on the normalized 3D points 
# by multiplying those points by your M matrix 
# and comparing the resulting the normalized 
# 2D points to the normalized 2D points given in the file.
ptsEstimate = np.matmul(M, threeDmatrix)
#  Remember to divide by the homogeneous value to get an inhomogeneous point. 
NoHomoEst = ptsEstimate.T[:, 0:2] / ptsEstimate.T[:, [2, 2]]
# The residual is just the distance 
# (square root of the sum of squared differences in u and v).  
residual = np.linalg.norm(NoHomoEst - np.asarray(twoD))
# print('estimated projection of pt %s is %s' % (twoD[-1], tuple(NoHomoEst[-1])))
print('\nresidual = ',residual)
print('\npts estimate: ', NoHomoEst)
# 1.b 
# normalized
setsize = range(1,12)
num_tests = 10
residuals = np.zeros((num_tests, len(setsize)))
(res_min, best_m) = (np.PINF, None)
for i in range(num_tests):
    for k_i, k in zip(range(len(setsize)), setsize):
        indexes = random.sample(range(0, len(threeD)), k)
        a = getAmatrix([threeD[i] for i in indexes], [twoD[i] for i in indexes])
        m = SVDtrick(a).reshape((3, 4))
        test_set = range(20)
        residual = calculate_avg_residual(m, [threeD[i] for i in test_set], [twoD[i] for i in test_set])
        (res_min, best_m) = (min(residual, res_min), m if residual < res_min else best_m)
        residuals[i][k_i] = residual

# plot results
plt.plot(setsize, residuals.mean(axis=0))
plt.xlabel("Set size K (used to compute M)")
plt.ylabel("Average Residual")
plt.xticks(setsize)
plt.show()
print("Residuals (each row represents avg residual for k size sets k = %s ) :\n%s" % (setsize, residuals))
print("Average residuals over all random tests  for k size sets k = %s: \n%s" % (setsize, residuals.mean(axis=0)))
print("Best M matrix and with residual (%s) \n%s" % (res_min, best_m))
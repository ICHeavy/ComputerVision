# ps2
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from disparity_ssd import disparity_ssd
from disparity_ncorr import disparity_ncorr

def SCALED(D):
    min_d = np.amin(D)
    max_d = np.amax(D)
    return (D - min_d)/(max_d - min_d)*255

# 1-a
# Read images
L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)

# # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)

# D_L = disparity_ssd(L, R)
# D_R = disparity_ssd(R, L)

# DL1A1 = np.copy(D_L)
# DR1A1 = np.copy(D_R)

# # TODO: Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
# # Note: They may need to be scaled/shifted before saving to show results properly

# # 1.A.
# D_L1_save = SCALED(DL1A1)
# D_R1_save = SCALED(DR1A1)
# cv2.imwrite('output/ps3-1-a-1.png', D_L1_save)
# cv2.imwrite('output/ps3-1-a-2.png', D_R1_save)


# TODO: Rest of your code here

# 2.A.
L2 = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
R2 = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1.0 / 255.0)

# D_L2 = disparity_ssd(L2, R2)
# D_R2 = disparity_ssd(R2, L2)

# DL2A1 = np.copy(D_L2)
# DR2A1 = np.copy(D_R2)

# D_L2_save = SCALED(DL2A1)
# D_R2_save = SCALED(DR2A1)
# cv2.imwrite('output/ps3-2-a-1.png', D_L2_save)
# cv2.imwrite('output/ps3-2-a-2.png', D_R2_save)

# # 3.A. 
# L3A = cv2.GaussianBlur(L2, (7, 7), 20)
# R3A = cv2.GaussianBlur(R2, (7, 7), 20)

# D_L3A = disparity_ssd(L3A, R3A)
# D_R3A = disparity_ssd(R3A, L3A)

# DL3AA1 = np.copy(D_L3A)
# DR3AA1 = np.copy(D_R3A)

# D_L3A_save = SCALED(DL3AA1)
# D_R3A_save = SCALED(DR3AA1)
# cv2.imwrite('output/ps3-3-a-1.png', D_L3A_save)
# cv2.imwrite('output/ps3-3-a-2.png', D_R3A_save)

# # 3.B.
# L3B = cv2.multiply(L2, np.ones(4), scale=1.1)
# R3B = np.copy(R2)

# D_L3B = disparity_ssd(L3B, R3B)
# D_R3B = disparity_ssd(R3B, L3B)

# DL3BA1 = np.copy(D_L3B)
# DR3BA1 = np.copy(D_R3B)

# D_L3B_save = SCALED(DL3BA1)
# D_R3B_save = SCALED(DR3BA1)
# cv2.imwrite('output/ps3-3-b-1.png', D_L3B_save)
# cv2.imwrite('output/ps3-3-b-2.png', D_R3B_save)


# 4.A.
L4 = np.copy(L2)
R4 = np.copy(R2)

D_L4 = disparity_ncorr(L4, R4)
D_R4 = disparity_ncorr(R4, L4)

DL4A1 = np.copy(D_L4)
DR4A1 = np.copy(D_R4)

D_L4_save = SCALED(DL4A1)
D_R4_save = SCALED(DR4A1)
cv2.imwrite('output/ps3-4-a-1.png', D_L4_save)
cv2.imwrite('output/ps3-4-a-2.png', D_R4_save)
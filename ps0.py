#LIBRARIES
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import random
# import skimage

#"C:\Users\simaj\Documents\School\CV\ps0_python_Sima_John"
############### GLOBALS ##############################
inPath = r'./input/'
outPath = "C:\\Users\\simaj\\Documents\\School\\CV\\ps0_python_Sima_John\\output"
photo1 = r"ps0-1-a-1.png"
photo2a = r"ps0-2-a-1.png"
photo2b = r"ps0-2-b-1.png"
photo2c = r"ps0-2-c-1.png"
photo3 =  r"ps0-3-a-1.png"
photo4b =  r"ps0-4-b-1.png"
photo4c =  r"ps0-4-c-1.png"
photo4d =  r"ps0-4-d-1.png"
photo4e =  r"ps0-4-e-1.png"
photo4f =  r"ps0-4-f-1.png"
photo5a =  r"ps0-5-a-1.png"
photo5b =  r"ps0-5-b-1.png"
photo5c =  r"ps0-5-c-1.png"
photo5e1 =  r"ps0-5-e-1.png"
photo5e2 =  r"ps0-5-e-2.png"
photo5f1 =  r"ps0-5-f-1.png"
photo5f2 =  r"ps0-5-f-2.png"



print("WELCOME ")
print("READING IMAGE...")
############### READING IN PHOTO ##############################
rawimg = cv2.imread(inPath + photo1, cv2.IMREAD_UNCHANGED)
h,w, ch = rawimg.shape
print("DIMENSIONS: ")
print(h,w)

################CHANGE TO SAVE DIRECTORY ###############
os.chdir(outPath)
print("OUTPUT BEFORE SAVE")  
print(os.listdir(outPath)) 

############### ISOLATE OG IMAGE ##############################
img1 = rawimg.copy()
img2a = cv2.cvtColor(rawimg, cv2.COLOR_BGR2RGB)

cv2.imwrite(photo1, img1)
cv2.imwrite(photo2a, img2a)
############### COLOR CHANNEL ISOLATION ##############################
blueChannel = rawimg[:,:,0]
BC = np.copy(blueChannel)

greenChannel = rawimg[:,:,1]
gmid = np.copy(greenChannel)
GC = np.copy(greenChannel)

redChannel = rawimg[:,:,2]
rout = np.copy(redChannel)
RC = np.copy(redChannel)


cv2.imwrite(photo2b, GC)
cv2.imwrite(photo2c, RC)
############### MATHING ############################################################
print(f"DOING SOME MATH...")


############### AVG AND STDDEV #############################################
calcavg, stdev = cv2.meanStdDev(greenChannel)
op = np.copy(greenChannel)
op = op.ravel()
max = np.max(op)
min = np.min(op)
print(f"AVG GREEN GUY VALUE = ", calcavg)
print(f"STD DEV GREEN GUY VALUE = ", stdev)
print(f"MAX GREEN GUY VALUE = ", max)
print(f"MIN GREEN GUY VALUE = ", min)


############### USE FOR FINDING HISTOGRAM OF SINGLE CHANNEL ###############
GChist = np.copy(greenChannel)
plt.hist(GChist.ravel(),256,[0,256])
plt.xlabel("GREEN HIST PRE OP")

plt.savefig(photo4b) # SAVE must come before show
plt.show()
############### BLUR #############################################

blurred = np.copy(greenChannel)
for i in range(blurred.shape[0]):
    for j in range(blurred.shape[1]):
        blurred[i][j] = (((greenChannel[i][j] - calcavg)/stdev)*10)+calcavg


############### BLUR HIST #############################################



BlurCopy = blurred.copy()
plt.hist(blurred.ravel(),256,[0,256])
plt.title("GREEN HIST POST OP BLUR")

plt.savefig(photo4d) # SAVE must come before save
plt.show()
cv2.imwrite(photo4c, blurred)
############### SHIFT ##############################
shift = 2
GCshift = np.copy(greenChannel)
for i in range(GCshift.shape[1] -1, GCshift.shape[1] - shift, -1):
    GCshift = np.roll(GCshift, -1, axis=1)
    GCshift[:, -1] = 0
plt.title("GREEN SHIFT")
# plt.imshow(GCshift)
# plt.show()


cv2.imwrite(photo4e, GCshift)
############### SUBTRACT ##############################
subbed = np.copy(GCshift)
gct = np.copy(greenChannel)
cv2.subtract(gct, GCshift,subbed,dtype=-1 )
plt.title("SUBBED")
# plt.imshow(subbed)
# plt.show()

cv2.imwrite(photo4f, subbed)
############### NOISE ############################################################
numpix = random.randint(1200,5000)
GCnoise = np.copy(GC)
BCnoise = np.copy(BC)


for i in range(numpix):
    #choose random pix to turn white
    ix = random.randint(0,h-1)
    iy = random.randint(0,w-1)
    GCnoise[ix,iy] = 255
    BCnoise[ix,iy] = 255

for i in range(numpix):
    #choose random pix to turn black
    ix = random.randint(0,h-1)
    iy = random.randint(0,w-1)
    GCnoise[ix,iy] = 0
    BCnoise[ix,iy] = 0


cv2.imwrite(photo5a, GCnoise)
cv2.imwrite(photo5c, BCnoise)
############### NOISE HISTOGRAMS ###############
# GREEN
GCnoiseH = np.copy(GCnoise)
plt.hist(GCnoiseH.ravel(),256,[0,256])
plt.title("NOISE HIST on GREEN")

plt.savefig(photo5b) # SAVE must come before show
plt.show()


############### FILTERING ##############################

median = cv2.medianBlur(GCnoise,5)
GCmedH = np.copy(median)
plt.hist(GCmedH.ravel(),256,[0,256])
plt.title("MEDIAN HIST on GREEN")

plt.savefig(photo5e2) # SAVE must come before show
plt.show()
cv2.imwrite(photo5e1, median)   

gaussBlur = cv2.medianBlur(GCnoise,5)
GCmedH = np.copy(gaussBlur)
plt.hist(GCmedH.ravel(),256,[0,256])
plt.title("GAUSS HIST on GREEN")

plt.savefig(photo5f2) # SAVE must come before show
plt.show()
cv2.imwrite(photo5f1, gaussBlur) 
############### DISPLAY ##############################
# Displaying the image

cv2.imshow('rgb', img1)
cv2.imshow('gbr', img2a)
cv2.imshow('green',GC)
cv2.imshow('red',RC)
cv2.imshow('blur', blurred)
cv2.imshow('shift', GCshift)
cv2.imshow('subbed', subbed)
cv2.imshow('Gnoise', GCnoise)
cv2.imshow('Bnoise', BCnoise)
cv2.imshow('median', median)
cv2.imshow('Bnoise', gaussBlur)
cv2.waitKey(0)
cv2.destroyAllWindows()


############### cropping and pasting due to overwrite logic ###############
print(f"SUCCESS! BEGIN CROP")
temp = np.zeros(gmid.shape)
greenMiddle = gmid[50:150, 50:150]
temp = rout[:,:]
temp[50:150,50:150] = greenMiddle
cropped = temp
#DISPLAY
cv2.imshow('middle',cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
 ### RESAVE ###
cv2.imwrite(photo3, cropped)

############### LIST FINAL CONTENTS ##############################
print("AFTER CROP SAVE:")  
print(os.listdir(outPath))

print("MISSION SUCCESS!") 


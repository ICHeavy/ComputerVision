#LIBRARIES
import cv2
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=sys.maxsize)

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

print("WELCOME ")
print("READING IMAGE...")
############### READING IN PHOTO ##############################
# GRABBING DIMENSIONS
# Using cv2.imread() method
# greyscale = 0
# color = 1
rawimg = cv2.imread(inPath + photo1)
height,width, nullguy = rawimg.shape
print("DIMENSIONS: ")
print(rawimg.shape)

################CHANGE TO SAVE DIRECTORY ###############
# DISPALY CONTENTS BEFORE AND AFTER 
os.chdir(outPath)
print("OUTPUT BEFORE SAVE")  
print(os.listdir(outPath)) 

############### ISOLATE OG IMAGE ##############################
img1 = rawimg.copy()
img2a = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)


############### COLOR CHANNEL ISOLATION ##############################
#(:,:,#)
#   [0 = blue, 1 = green, 2 = red,3 = transparent]
greenChannel = img1[:,:,1].copy()
redChannel = img1[:,:,2].copy()



############### MATHING ############################################################
print(f"DOING SOME MATH...")

############### AVERAGE ###############
#returns (red, green, blue, tranparent)
shift = 2
for i in range(greenChannel.shape[1] -1, greenChannel.shape[1] - shift, -1):
    greenChannel = np.roll(greenChannel, -1, axis=1)
    greenChannel[:, -1] = 0

cv2.imshow('image', greenChannel)
cv2.waitKey()

################BELOW WORKS BUT ABOVE IS BETTER###############
# greensum = total = oTempAvg = 0
# oTempList = np.zeros(rawimg.shape)
# for i in range(height):
#     for j in range(width): 
#         #
#         # print(" green value at", i, ",", j, " is: ", rawimg.item(i,j,1))
#         greensum += rawimg.item(i,j,1)
#         total += 1
# greenavg = greensum/total
# print(f"CALC AVG GREEN GUY VALUE = ", greenavg)
###########################################################################

#PROABABLY A OPENCV STDEV F(X) TOO
#THERE IS! wait it gets better

############### AVG AND STDDEV #############################################
calcavg, stdev = cv2.meanStdDev(greenChannel)
print(f"AVG GREEN GUY VALUE = ", calcavg)
print(f"STD DEV GREEN GUY VALUE = ", stdev)

############### pixel operations ###############
#nested lists
# Subtract the mean from all pixels, 
# then divide by standard deviation, 
# then multiply by 10 (if yourimage is 0 to 255) 
# Now add the mean back in.

# For Image shape: image.shape
# For getting a pixel: image[row][col]
# For setting a pixel: image[row][col] = [r,g,b]

############### USE FOR FINDING HISTOGRAM OF ALL CHANNELS ###############
color = ('r', 'b', 'g')

greenGoblin = np.zeros(img1.shape)
for i in range(width):
    for j in range(height): #go thru every pixel
        tempPix = img1[i,j] #grab pixel list
        tempG = tempPix[1] #isolate green element
        greenGoblin[i,j] = [0,tempG,0]

print(greenGoblin)
# enumerate runs backward
# and also gbr not rgb
# for count, element in enumerate(list):
for i,col in enumerate(color):
    histr = cv2.calcHist([img1],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig(photo4b) #must come before save
plt.show()



# plt.hist(greenChannel.ravel(),256,[0,256])
# plt.savefig(photo4c) #must come before save
# plt.show()


# greenGoblin = np.zeros(img1.shape)
# for i in range(200):
#     for j in range(200):
#         x = (((greenChannel[i][j] - calcavg)/stdev)*10)+calcavg
#         greenGoblin[i, j] = [x, x, x]
      

# #greenGoblin = greenGoblin.astype(int)
# print("OG:")
# print(greenChannel)
# print("gC:")
# print(greenGoblin)
# greenGoblinTemp = greenGoblin.copy()

# plt.hist(greenGoblinTemp.ravel(),256,[1,256])
# plt.savefig(photo4c) #must come before save
# plt.show()


############### SINGLE CHANNEL HISTOGRAM ###############
#flatten array of single channel (not effeicent method for full color, use above)
############### SINGLE CHANNEL HISTOGRAM ###############
# plt.hist(greenChannel.ravel(),256,[0,256])
# plt.savefig(photo4b) #must come before save
# plt.show()

############### TO RESIZE ##############################
#dimensions = (400,800)
#resized = cv2.resize(rawimg, dimensions, interpolation = cv2.INTER_AREA)

############### DISPLAY ##############################
print(f'DISPLAYING...')
# Displaying the image
cv2.imshow('rgb', img1)
cv2.imshow('gbr', img2a)
cv2.imshow('green',greenChannel)
cv2.imshow('red',redChannel)
cv2.imshow('altered', greenGoblin)
cv2.waitKey(0)
cv2.destroyAllWindows()

############### SAVING ##############################
#saving these prior to crop due to some overwrite i cant pin down
print(f"SAVING...")
cv2.imwrite(photo1, img1)
cv2.imwrite(photo2a, img2a)
cv2.imwrite(photo2b, greenChannel)
cv2.imwrite(photo2c, redChannel)


############### cropping and pasting due to overwrite logic ###############
print(f"SUCCESS! BEGIN CROP")
temp = np.zeros(rawimg.shape)
greenMiddle = greenChannel[50:150, 50:150]
temp = redChannel[:,:]
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


import cv2
import numpy as np
import matplotlib.pyplot as plt
import winsound

inPath = r'./input/'
outPath = r'./output/'

photo1 = r"ps2-input0.png"
photo2 = r"ps2-input0-noise.png"
photo3 = r"ps2-input1.png"
photo4 = r"ps2-input2.png"
# practicephoto = r"practiceimg.png"

outphoto1 = r"ps2-1-a-1.png"
outphoto2a = r"ps2-2-a-1.png"
outphoto2b = r"ps2-2-b-1.png"
outphoto2c = r"ps2-2-c-1.png"
outphoto3a = r"ps2-3-a-1.png"
outphoto3b1 = r"ps2-3-b-1.png"
outphoto3b2 = r"ps2-3-b-2.png"
outphoto3c1 = r"ps2-3-c-1.png"
outphoto3c2 = r"ps2-3-c-2.png"
outphoto4a = r"ps2-4-a-1.png"
outphoto4b = r"ps2-4-b-1.png"
outphoto4c1 = r"ps2-4-c-1.png"
outphoto4c2 = r"ps2-4-c-2.png"
outphoto5a1 = r"ps2-5-a-1.png"
outphoto5a2 = r"ps2-5-a-2.png"
outphoto5a3 = r"ps2-5-a-3.png"
outphoto5b = r"ps2-5-b-1.png"
outphoto6a = r"ps2-6-a-1.png"
outphoto6c = r"ps2-6-c-1.png"
outphoto7a = r"ps2-7-a-1.png"
outphoto8a = r"ps2-8-a-1.png"

compare = "compare.png"

# img1 = Image.open(inPath+photo1)

# #go to grayscale
# img1.convert("L")
# #apply Laplac kernal
# img1res = img1.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 0))
# img1res.save(outPath+outphoto1)

# # each point votes for compatible lines
# #   record all lines where theres edges
# #   look for lnes 



#   CANNY EDGE DETECTION
def unCanny(img, outphoto1):
    edges = cv2.Canny(img,100,200)
    cv2.imwrite(outPath+outphoto1, edges)
    # cv2.imshow('Canny Edges', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return edges

def hough_lines_acc(img): 
    h,w = img.shape
    #  bc trianlges
    D = np.ceil(np.sqrt(h*h + w*w)) # *********************

    # generate theta [-90:90]
    # THETA IN 180 DEG RANGE      
    theta_arr = np.deg2rad(np.arange(-90.0, 90.0,1))
    rho_arr = np.arange(-D,D+1, 1)
    numThetas = len(theta_arr)
    
    # acc array of doubles (H) has:
    # X axis -  theta [-90:90]
    # Y axis - rho (dist from origin)
    H = np.zeros((len(rho_arr), numThetas ), np.uint64)
    iy, ix = np.nonzero(img)  # (row, col) indexes of edges, but flipped
    #  for traveling down each y value
    # grab index of each nonzero value (of each edge)
    for i in range(len(ix)):
        x = ix[i]
        y = iy[i]
        # go through each theta 
        # calculate the distance from that point to origin
        # inc the accumulator arr

        for j in range(numThetas):
            rho = int(x*np.cos(theta_arr[j]) + y*np.sin(theta_arr[j]) + D)
            H[rho, j] += 1
        


    print("\n 100.0 %\n !!! FINALLY DONE !!!")
    
    winsound.Beep(800, 2000) # so i can go do other thing
    return [H, theta_arr, rho_arr]


def hough_peaks(H, Npeaks, border = 5):
    peaks = []
    Htemp = np.copy(H)
    # loop through number of peaks
    for i in range(Npeaks):
        print(i)
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

def hough_lines_draw(img, peaks, theta_arr, rho_arr):
    
    for i in range(len(peaks)):
        print("drawing line: # ", i)
        
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
    
def hough_circles_acc(img, r):
    # create new H
    H = np.zeros(img.shape)
    Htemp = np.zeros(img.shape)

    # get coordinates of the image and map to lin index 
    # go through all the edge points, flip x/y bc Q4
    # the hasmap index is 
    for (y, x), is_edge in np.ndenumerate(img):
        if is_edge:
            print(y, " hca\n")
            # draw a circle on in Htmep and add its vote to H
            temp = Htemp.copy()
            cv2.circle(temp, (x, y), r, 1, thickness=1)
            H += temp

    # all circles are drawn onto H, 
    # the places where all these circles overlap
    # will be the location of the circle as the votes from surrounding 
    # points will distinguish it
    return H

def find_circles(img, radRng):
    # create 2D arr of coordinates 
    centers = np.empty((0, 2), int)
    radii = []
    # grab each lin index of coordinate via hashmap
    # i.e. [20,50] - list [20,50), indexed [0,29)
    for i, rad in enumerate(np.arange(radRng[0], radRng[1])):
        # travel through the range of radii
        oH = hough_circles_acc(img, rad)
        print('--------------------------- Radius: ', rad, ' -----------------------------')
        centerPeaks, oHtmep = hough_peaks(oH, 5)
        # grab centers off each radius, add those to both lists as you go
        centers = np.concatenate((centers, centerPeaks), axis=0)
        radlist = [rad]
        radii += radlist*len(centerPeaks)

    return centers, np.array(radii)


#read in BW
#  QUESTION 1
# img0 = cv2.imread(inPath+photo1)
# img1 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
# cv2.imshow('OG Image', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# edges = unCanny(img1,outphoto1)

# QUESTION 2
# [H, theta_arr, rho_arr] = hough_lines_acc(edges)
# cv2.imwrite(outPath+outphoto2a, H)
# peaks, Htmp = hough_peaks(H, 10)
# cv2.imwrite(outPath+outphoto2b, Htmp)
# lines = hough_lines_draw(img1, peaks, theta_arr, rho_arr)
# cv2.imwrite(outPath+outphoto2c, lines)

# QUESTION 3
# img2 = cv2.imread(inPath+photo2)
# blurred = cv2.GaussianBlur(img2, (5, 5), 20) # for noisy
# cv2.imwrite(outPath+outphoto3a, blurred)
# cv2.imshow('blurred Image', blurred)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# noisyEdges = unCanny(img2, outphoto3b1)
# smoothedEdges = unCanny(blurred, outphoto3b2)

# [Hnoise, theta_arrnoise, rho_arrnoise] = hough_lines_acc(smoothedEdges)
# peaksnoise, Htmpnoise = hough_peaks(Hnoise, 10)
# cv2.imwrite(outPath+outphoto3c1, Htmpnoise)
# linesnoise = hough_lines_draw(blurred, peaksnoise, theta_arrnoise, rho_arrnoise)
# cv2.imwrite(outPath+outphoto3c2, linesnoise)

# QUESTION 4
# img3 = cv2.imread(inPath+photo3)
# img4 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)
# blurguy = cv2.GaussianBlur(img4, (5, 5), 20) # for noisy
# cv2.imwrite(outPath+outphoto4a, blurguy)
# cv2.imshow('blurred REAL', blurguy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# penEdges = unCanny(blurguy, outphoto4b)
# [penH, thetaPen, rhoPen] = hough_lines_acc(penEdges)
# penPeaks, penHtemp = hough_peaks(penH, 6)
# cv2.imwrite(outPath+outphoto4c1, penHtemp)
# penLines = hough_lines_draw(blurguy, penPeaks, thetaPen, rhoPen)
# cv2.imwrite(outPath+outphoto4c2, penLines)

# # QUESTION 5
# cv2.imwrite(outPath+outphoto5a1, blurguy)
# penEdges5 = unCanny(blurguy, outphoto5a2)
# # [penH5, thetaPen5, rhoPen5] = hough_lines_acc(penEdges5)
# # penPeaks, penHtemp = hough_peaks(penH5, 10)
# circleH = hough_circles_acc(penEdges5, 10)
# oPenPeaks, oHtemp = hough_peaks(circleH, 10, 15)
# cv2.imwrite(outPath+outphoto5a3, oHtemp)
# # cv2.imshow('hough circle plot', circleH)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# oCenters, radii = find_circles(penEdges5, [20,50])
# for i, center in enumerate(oCenters):
#         x = int(center[1])
#         y =  int(center[0])
#         cv2.circle(img3, (x,y), radii[i], (0, 255, 0), thickness=1)
    
# cv2.imwrite(outPath+outphoto5b, img3)


# QUESTION 6
img5 = cv2.imread(inPath+photo4)
img5t = cv2.cvtColor(img5, cv2.COLOR_RGB2GRAY)
blur6 = cv2.GaussianBlur(img5, (5, 5), 2)
edges6 = unCanny(blur6, "6edges.png")
[H6, T6, R6] = hough_lines_acc(edges6)
peaks6, H6res = hough_peaks(H6, 8, 10)
res6 = hough_lines_draw(blur6, peaks6, T6, R6 )
cv2.imwrite(outPath+outphoto6a, res6)

# QUESTION 7
oH7 = hough_circles_acc(edges6, 25)
o7peaks, oH7temp = hough_peaks(oH7, 10)
cv2.imwrite(outPath+"hough7.png", oH7temp)

oCenters7, radii7 = find_circles(edges6, [25,50])
for i, center in enumerate(oCenters7):
        x = int(center[1])
        y =  int(center[0])
        cv2.circle(blur6, (x,y), radii7[i], (0, 255, 0), thickness=1)
    
cv2.imwrite(outPath+outphoto7a, blur6)

print("Done")

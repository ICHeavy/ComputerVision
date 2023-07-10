import cv2
import os
import numpy as np

inPath = r'./input/'
outPath = r'./output/'



def arrows(u, v, lenguy, size, color= (255,255,255)):
     # from quiver
    #  create empty arr of proper size
    output = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)
    for y in range(0, v.shape[0], lenguy):
        for x in range(0, u.shape[1], lenguy):
            # draw line from original pos to shifted pos
            # add circle at the end of line
            # pixels so it looks like arrows
            cv2.line(output, (x, y), (x + int(u[y, x] * size), y + int(v[y, x] * size)), color, 1)
            cv2.circle(output, (x + int(u[y, x] * size), y + int(v[y, x] * size)), 1, color, 1)
    return output


def LKopticFlow(img1, img2, k):
    # Convolution is just a normalized sum.
    # return U and V displacements along x & y respectively
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5, scale=1./8)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5, scale=1./8)
    It = img2- img1
    # use Sobel kernal to get gradient respectively, k = 5 from assgn, type and scale from documentation

    window =  cv2.getGaussianKernel(k, 3, cv2.CV_64F)

    # A^T*A = M = [sum(Ix*Ix) sum(Ix*Iy)]  
    #             [sum(Ix*Iy) sum(Iy*Iy)]

    # [Ix(1,1) Iy(1,1)]  [u]        [iT(1,1)]
    # ...       ...
    # [Ix(k,l) Iy(k,l)]         =   [iT(K,L)]
    # ...       ...
    # [Ix(n,n) Iy(n,n)]  [V]        [iT(n,n)]

    # Ix(k,l)u + Iy(k,l)v + It(k,l) = 0 
    # where k,l is pix in window

    # u = (A^T*A)^-1 * A^T*B
    
    
    IxIt = cv2.filter2D(Ix*It, cv2.CV_64F, window)*(-1)
    IyIt = cv2.filter2D(Iy*It, cv2.CV_64F, window)*(-1)
    IxIx = cv2.filter2D(Ix*Ix, cv2.CV_64F, window)
    IyIy = cv2.filter2D(Iy*Iy, cv2.CV_64F, window)
    IxIy = cv2.filter2D(Ix*Iy, cv2.CV_64F, window)

   

    # A^T*A must be invertable => det(A^TA) != 0
    det = (IxIx * IyIy - IxIy ** 2) ** (-1)
    det[det == np.inf] = 0

    u = det*(IyIy*IxIt-IxIy*IyIt)
    v = det*((IxIx*IyIt)-(IxIy*IxIt))

    return u,v

def blurMe(img,size):
    img1=img.copy()
     # img = cv2.GaussianBlur(img, (blur_size, blur_size), blur)
    img = img.astype(np.float32)
    img = cv2.bilateralFilter(img, -1, size/2, size)
    return img


noshift = cv2.imread(os.path.join('input/TestSeq', 'Shift0.png'), 0) / 255.0
shiftR2 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR2.png'), 0) / 255.0
shiftR5U5 = cv2.imread(os.path.join("input", 'TestSeq','ShiftR5U5.png'), 0) / 255.
shiftR10 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR10.png'), 0) / 255.0
shiftR20 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR20.png'), 0) / 255.0
shiftR40 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR40.png'), 0) / 255.0
#  R2
u,v = LKopticFlow(noshift.copy(), shiftR2, 25)
res1 = arrows(u,v, 5,20)
cv2.imwrite(os.path.join("output", "ps4-1-a-1.png"), res1)

# R5U5
blur = blurMe(shiftR5U5,6)
og = blurMe(noshift,6)
og10 = og.copy()
u1,v1 = LKopticFlow(og, blur, 40)
res2 = arrows(u1,v1,5,20)
cv2.imwrite(os.path.join("output", "ps4-1-a-2.png"), res2)

# R10
blur10 = blurMe(shiftR10,6)

u10,v10 = LKopticFlow(og10, blur10, 30)
res3 = arrows(u10,v10,5,20)
cv2.imwrite(os.path.join("output", "ps4-1-b-1.png"), res3)

# R20 
blur20 = blurMe(shiftR20,6)
u20,v20 = LKopticFlow(og, blur20, 30)
res4 = arrows(u20,v20,5,20)
cv2.imwrite(os.path.join("output", "ps4-1-b-2.png"), res4)

# R40
blur40 = blurMe(shiftR40,6)
u40,v40 = LKopticFlow(og, blur40, 30)
res5 = arrows(u40,v40,5,20)
cv2.imwrite(os.path.join("output", "ps4-1-b-3.png"), res5)


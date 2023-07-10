import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from skimage.feature import hog 
import os 
import cv2
from sklearn.svm import LinearSVC 

def getHOGVector(image_name): 
  
  #read the image 
  image = plt.imread(image_name) 
  
  #compute the HOG features 
  features, hog_image = hog(image,cells_per_block=(2, 2),visualize=True,channel_axis=2) 
  
  #return the features and the corresponding hog image 
  return features, hog_image 

def getXnY(file):
    xtrain = np.empty(shape=(0, 1188)) 
    ytrain = np.empty(shape=(0,))
    for image_name in os.listdir('input/p1/'+file+'_imgs'): 
        #get the label 
        label = int(image_name[0]) 

        #read the image 
        image = plt.imread('input/p1/train_imgs/' + image_name) 
    
        #compute the HOG features 
        features, hog_image = hog(image,cells_per_block=(2, 2),visualize=True,channel_axis=2) 
    
        #append
        xtrain = np.append(xtrain, np.array([features]), axis=0) 
        ytrain = np.append(ytrain, np.array([label]), axis=0) 

    #print the shapes of X_train and y_train 
    print(file,' shape:', xtrain.shape) 
    print(file, ' shape:', ytrain.shape)
    return xtrain, ytrain

def classifyMe(x, y):
    clf = LinearSVC() 
    #train the classifier using the X_train and y_train values 
    clf.fit(x,y) 
    return clf

def trainModel(clf):
    #loop through the testing images 
    k = 1
    for image_name in os.listdir('input/p1/test_imgs'): 
        #read the image 
        print(image_name, "\n")
        image = plt.imread('input/p1/test_imgs/' + image_name) 
    
        #find the maximum score 
        max_score = -1 
        max_i = 0 
        max_j = 0 
    
        #loop through the image 
        for i in range(image.shape[0] - 31): 
            for j in range(image.shape[1] - 95): 
                #extract the window 
                window = image[i:i+32, j:j+96] 
    
                #compute the HOG features 
                features, hog_image = hog(window, orientations=9,pixels_per_cell=(8, 8),cells_per_block=(2, 2),visualize=True,
                                        block_norm='L2-Hys', channel_axis=2) 
    
                #get the score 
                score = clf.decision_function(np.array([features])) 
    
                #update the maximum score 
                if score > max_score: 
                    max_score = score 
                    max_i = i 
                    max_j = j 
    
        #check if the score is greater than 1.2 
        if max_score > 1.2: 
            #draw a rectangle 
            cv2.rectangle(image, (max_j, max_i), (max_j+32, max_i+96), (0, 0, 255), 2) 
        fn = 'output/ps7-1-d-'+ str(k) +'.png'
        cv2.imwrite(fn, image)
        k+=1

# 1a
#test this on 'input/p1/car.jpg' 
features, hog_img = getHOGVector('input/p1/car.jpg') 

#display the hog image 
plt.imshow(hog_img) 
plt.savefig('ps7-1-a.png') 
plt.show()

# 1b 
test = "test"
train = "train"
xtrain, ytrain = getXnY(train)

# 1c 
classy = classifyMe(xtrain, ytrain)

# 1d
xtest,_ = getXnY(test)
print("almost")
trainModel(classy)
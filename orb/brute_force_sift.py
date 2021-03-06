import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img1 = cv.imread('answer.png',0)       
img2 = cv.imread('main3.jpg',0)
# Initiate SIFT detector
sift =  cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)



cv.imwrite('brute_force_result_sift.jpg', img3)
plt.imshow(img3),plt.show()
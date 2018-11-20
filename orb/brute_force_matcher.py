#Brute-Force 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('answer.png',0)       
img2 = cv.imread('main3.jpg',0)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
#img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
cv.imwrite('brute_force_result.jpg', img3)
plt.imshow(img3),plt.show()
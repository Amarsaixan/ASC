import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('main3.jpg',0)

# Initiate STAR detector
orb = cv2.ORB_create()        # Initiate SIFT detector


# find the keypoints with ORB
kp = orb.detect(img,None)
outImage =None
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
#kp, des = orb.detectAndCompute(img, None)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,outImage,color=(0,255,0), flags=0)
#plt.imshow(img2),plt.show()
cv2.imwrite('orb_result.png',img2)
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread("main3.jpg", cv2.IMREAD_GRAYSCALE)
 
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
 
orb = cv2.ORB_create(nfeatures=1500)
 
keypoints, descriptors = orb.detectAndCompute(img, None)
 
img = cv2.drawKeypoints(img, keypoints, None)
cv2.imwrite('orb_keypoint_result.png',img)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
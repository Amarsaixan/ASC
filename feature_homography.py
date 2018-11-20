import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
MIN_MATCH_COUNT = 10
cap = cv2.VideoCapture(0)
img1 = cv2.imread('amaraa.jpeg',0)          # queryImage
#img2 = cv2.imread('main.jpg',0) # trainImage
if img1 is None:
	print("img1 not found")
	sys.exit(0)


# Initiate SIFT detector
#sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()


# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)

while (True):
	ret, img2 = cap.read()
	
	kp2, des2 = sift.detectAndCompute(img2,None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
	    if m.distance < 0.7*n.distance:
	        good.append(m)
	if len(good)>MIN_MATCH_COUNT:
	    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	    matchesMask = mask.ravel().tolist()

	    h,w = img1.shape
	    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	    dst = cv2.perspectiveTransform(pts,M)

	    img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,0,255), cv2.LINE_AA)

	else:
	    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
	    matchesMask = None
	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
	                   singlePointColor = None,
	                   matchesMask = matchesMask, # draw only inliers
	                   flags = 2)

	img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

	cv2.imshow('frame',img2)
	#plt.imshow(img3, 'gray'),plt.show()
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
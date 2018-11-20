import numpy as np
import cv2
from matplotlib import pyplot as plt
def maskFrame(mask):
    rs,cs, k = mask.shape
    for x in range(0,rs-1):
        for y in range(0,cs-1):
            if mask[x,y][0]<=155:
                mask[x,y] = 0
            else:
                mask[x,y] = 255 
    return mask
MIN_MATCH_COUNT = 10

img1 = cv2.imread('answer.png',0)          # queryImage
img2 = cv2.imread('main4.jpg',0) # trainImage
cap = cv2.VideoCapture(0)
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

kp2, des2 = sift.detectAndCompute(img2,None)
matches = flann.knnMatch(des1,des2,k=2)
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
    #print(str(len(good)))
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
    print(dst)
        #for t in range(len(dst)):
        #    print(dst.get(t))
            #cv2.circle(img2,(100,100), 5, (0,0,255), -1)
    pts1 = np.float32(dst)
    pts2 = np.array([
            [0, 0],
            [h - 1, 0],
            [h - 1, w - 1],
            [0, w - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(pts1,pts2)
    ff = cv2.warpPerspective(img2,M,(h,w))
    ff =  cv2.flip( ff, 1)
    cv2.imwrite('mff.png',ff)

cv2.imshow('img2', img2)

cap.release()
cv2.destroyAllWindows()

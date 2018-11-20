import cv2
import numpy as np
import random
#random.randint(1,255),random.randint(1,255),random.randint(1,255)
img = cv2.imread('answer.png')

#cv2.rectangle(img,(0, 0),(100, 200),(0,255,0),4)
rows,cols,ret = img.shape	
bagana = int(cols/12)
mor = int(rows/35)
print(bagana)
print(mor)
for i in range((mor*6),rows,mor):
	for j in range((bagana*2),cols,bagana):
			#print(str(i)+","+str(j))
			#crop_img = ff[i:i+mor, j:j+bagana]
			print(":")
			print(i)
			print(j)
			cv2.line(img,(j,i),(j, i+mor),(255,0,0),5)
			cv2.line(img,(j,i),(j+bagana, i),(255,0,0),5)
			cv2.line(img,(j, i+mor),(j+bagana, i+mor),(255,0,0),5)
			cv2.line(img,(j+bagana, i),(j+bagana, i+mor),(255,0,0),5)
			#cv2.rectangle(img,(j, i),(bagana, mor),(0,255,0),4)
cv2.imwrite('answer_problem.png',img)
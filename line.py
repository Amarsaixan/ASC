import numpy as np
import cv2
img = cv2.imread('answer.jpg', 0)
cv2.rectangle(img,(0,0),(510,128),(255,255,0),6)
while True:
	cv2.imshow('frame', img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
img.release()
cv2.destroyAllWindows()
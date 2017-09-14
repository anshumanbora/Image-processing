import numpy as np
import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
img = cv2.imread('./images/1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# equalize the histogram of the Y channel
#img[:,:,0] = cv2.equalizeHist(img[:,:,0])
# convert the YUV image back to RGB format
#img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

#faces=[]
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#print(type(faces))
count=0
imgFace=[]

if faces==():
	print(" No face detected ")
	sys.exit()

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    if count==0:
    	imgFace=roi_color	
    	break

redLower = (0, 80, 80)
redUpper = (100, 255, 255)

hsv = cv2.cvtColor(roi_color,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, redLower, redUpper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

cv2.imshow('hsv',hsv)

[im2,cnts,hierarchy]= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
contour_id = 0
border_thickness = 2
border_color = (185, 115, 72)
cv2.drawContours(roi_color, cnts, -1, border_color, border_thickness)
#print(len(cnts))
#Now you again draw contour but with thickness = -1 and color = Core color
border_thickness = -1
core_color = (225, 141, 98)
ellipse = cv2.fitEllipse(cnts[0])
#cv2.ellipse(roi_color,ellipse,(0,255,0),2)
# cv2.drawContours(img, cnts, contour_id, core_color, border_thickness)

cv2.imshow('img',img)
cv2.imshow('img1',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()	
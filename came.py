import cv2

img = cv2.imread('im1.jpeg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_rect = h.detectMultiScale(
	gray_img, scaleFactor=1.1,
	minNeighbors=10)
for (x, y, w, h) in faces_rect:
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv2.imshow('Detected faces', img)
cv2.waitKey(0)



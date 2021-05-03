import dlib
import cv2 as cv
import numpy as np
import sys

cap = cv.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def createbox(img, points, scale=3, masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv.fillPoly(mask, [points], (255, 255, 255))
        img = cv.bitwise_and(img, mask)
        # cv.imshow('Mask',img)
    # if cropped:
    #     box = cv.boundingRect(points)
    #     x, y, w, h = box
    #     imgcrop = img[y:y + h, x:x + w]
    #     imgcrop = cv.resize(imgcrop, (0, 0), None, scale, scale)
    #     return imgcrop
    # else:
        return mask

while True:

    success, img = cap.read()
    img = cv.resize(img,(0,0),None,1,1)
    imgOriginal = img.copy()
    imggray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = detector(imggray)

    for face in faces:
      landmarks = predictor(imggray,face)
      points = []

      for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append([x,y])
      points = np.array(points)
      lip = createbox(img,points[48:68],3,masked=True,cropped=False)
      colorlip = np.zeros_like(lip)
      colorlip[:] = (153,0,157)
      colorlip = cv.bitwise_and(lip,colorlip)
      colorlip = cv.GaussianBlur(colorlip,(7,7),10)
      colorlip = cv.addWeighted(imgOriginal,1,colorlip,0.4,0)
      #cv.imshow("Normal",imgOriginal)
      cv.imshow("Color",colorlip)

    key = cv.waitKey(1)
    if key%256 == 27:
     break

cv.waitKey(0)
cap.release()
cv.destroyAllWindows()
import dlib
import cv2 as cv
import numpy as np

# Detects the face from the input such as Webcam, or Read method.
detector = dlib.get_frontal_face_detector()

# Predicts the 68 point landmark from the dat file.  And applies the face-geometrics to the input image
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img = cv.imread("1.jpg")
img = cv.resize(img, (0, 0), None, 0.5, 0.5)
imgOriginal = img.copy()

# Applying GrayScale conversion to the input image, since the detector method accepts only the grayscale format.
imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = detector(imggray)
cv.imshow("Normal", imgOriginal)


def createbox(img, points):
    mask = np.zeros_like(img)
    mask = cv.fillPoly(mask, [points], (255, 255, 255))
    return mask


for face in faces:

    # Inputs the grayscale image to the predictor, along with detected facial informations.
    landmarks = predictor(imggray, face)
    points = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        # Appending the retrieved co-ordrinates from the face, to a list.
        points.append([x, y])

    points = np.array(points)

    # Invoking createbox to filter the lip area
    lip = createbox(img, points[48:68])
    colorlip = np.zeros_like(lip)    #Creating a mask image from the original one, and extracting the lip area
    colorlip[:] = 153, 0, 167
    colorlip = cv.bitwise_and(lip, colorlip)   #Applies the input color to the specified region.
    colorlip = cv.GaussianBlur(colorlip, (7, 7), 10)   #Reduces the noise in the refined image
    colorlip = cv.addWeighted(imgOriginal, 1, colorlip, 0.3, 0) #Merges the Original image, with the extracted colored lip
    cv.imshow("Normal", imgOriginal)
    cv.imshow("Colored Lip", colorlip)

cv.waitKey(0)

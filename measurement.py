import cv2
import numpy as np
import utilis

capture = cv2.VideoCapture(0)
capture.set(10,160)
capture.set(3,1920)
capture.set(4,1080)
scale = 3
paperW=210*scale
paperH=297*scale


while True:
    success, image = capture.read()

    imageContours,conts = utilis.getContour(image,minimumArea=50000,filter=4)

    if len(conts)!=0:
        biggest = conts[0][2]

        imgWarp = utilis.warp(image, biggest, paperW, paperH)

        imageContours2, conts2 = utilis.getContour(imgWarp, minimumArea=2000, filter=4, cannyThresh=[50, 50], draw=False)
        if len(conts) !=0:
            for object in conts2:
                cv2.polylines(imageContours2,[object[2]],True,(0,255,0),2)
                nPoints = utilis.order(object[2])
                newWidth = round((utilis.Distance(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
                newHeight = round((utilis.Distance(nPoints[0][0]//scale,nPoints[2][0]//scale)/10),1)
        cv2.imshow('A4', imageContours2)

    cv2.imshow('Result',image)

    cv2.waitKey(1)


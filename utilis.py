import cv2
import numpy as np
def getContour(image,cannyThresh=[100,100],showThresh=False,minimumArea=1000,filter=0,draw=False):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageBlur = cv2.GaussianBlur(imageGray, (5,5),1)
    imageCanny = cv2.Canny(imageBlur,cannyThresh[0],cannyThresh[1])
    kernel = np.ones((5,5))
    imageDilation = cv2.dilate(imageCanny,kernel,iterations=3)
    imageFinal = cv2.erode(imageDilation,kernel,iterations=2)
    if showThresh:cv2.imshow('canny output',imageFinal)

    contours,hiearchy = cv2.findContours(imageFinal,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    goodContours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area>minimumArea:
            perimeter = cv2.arcLength(i,True)
            cornerPoints = cv2.approxPolyDP(i,0.02*perimeter,True)
            boundary = cv2.boundingRect(cornerPoints)
            if filter > 0:
                if len(boundary)==filter:
                    goodContours.append([len(boundary),area,cornerPoints,boundary,i])
            else:
                goodContours.append([len(boundary),area,cornerPoints,boundary,i])
    goodContours = sorted(goodContours,key= lambda x:x[1],reverse=True)

    if draw:
        for contours in goodContours:
            cv2.drawContours(image,contours[4],-1,(0,0,255),3)

    return image, goodContours

def order(points):
    print(points.shape)
    pointsNew = np.zeros_like(points)
    points=points.reshape((4,2))
    add = points.sum(1)
    pointsNew[0]=points[np.argmin(add)]
    pointsNew[3] = points[np.argmax(add)]
    diff = np.diff(points,axis=1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]
    return pointsNew

def warp(image,points,width,height,pad=20):
    points = order(points)
    points1 = np.float32(points)
    points2 = np.float32([0,0],[width,0][0,height],[width,height])
    matrix = cv2.getPerspectiveTransform(points1,points2)
    imgWarp = cv2.warpPerspective(image,matrix,(width,height))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]

    return imgWarp


def Distance(points1,points2):
    return ((points2[0]-points1[0])**2 + (points2[1]-points1[1])**2)**0.5
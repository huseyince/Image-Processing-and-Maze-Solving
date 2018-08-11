#Solving maze with morphological transformation
"""
usage:Solving maze with morphological transformation
needed module:cv2/numpy/sys
ref:
1.http://www.mazegenerator.net/
2.http://blog.leanote.com/post/leeyoung/539a629aab35bc44e2000000
@author:Robin Chen
"""
import cv2
import numpy as np
import sys
def SolvingMaze(image):
#load an image
    try:
        img = cv2.imread(image)
    except Exception,e:
        print 'Error:can not open the image!'
        sys.exit()
#show image
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('maze_image',img)
#convert to gray
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#show gray image
    #cv2.imshow('gray_image',gray_image)

#convert to binary image
    retval,binary_image = cv2.threshold(gray_image, 10,255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('binary_image',binary_image)

    _, contours,hierarchy  = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 2:
        sys.exit("This is not a 'perfect maze' with just 2 walls!")
        
    h, w, d = img.shape

#The first wall
    path = np.zeros((h,w),dtype = np.uint8)#cv2.CV_8UC1
    cv2.drawContours(path, contours, 0, (255,255,255),-1)#cv2.FILLED
    #cv2.imshow('The first wall',path)

#Dilate the wall by a few pixels
    kernel = np.ones((19, 19), dtype = np.uint8)
    path = cv2.dilate(path, kernel)
    #cv2.imshow('Dilate the wall by a few pixels',path)

#Erode by the same amount of pixels
    path_erode = cv2.erode(path, kernel);
    #cv2.imshow('Erode by the same amount of pixels',path_erode)

#absdiff
    path = cv2.absdiff(path, path_erode);
    #cv2.imshow('absdiff',path)

#solution
    channels = cv2.split(img);
    channels[0] &= ~path;
    channels[1] &= ~path;
    channels[2] |= path;

    dst = cv2.merge(channels);
    cv2.imwrite("solution.png", dst);
    cv2.imshow("solution", dst);
#waiting for any key to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image = sys.argv[-1]
    SolvingMaze(image)

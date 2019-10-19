
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

count = 0

cap = cv.VideoCapture('testvideo1.mp4')
success, image = cap.read()

while success:
    cv.imwrite('frames/frame_{}.png'.format(count), image)
    success, image = cap.read()
    print('Read a new frame: {}'.format(success))
    count += 1

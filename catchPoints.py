from turtle import shape
import numpy as np
import cv2 as cv2

def getSelectedPoints(img):
    #https://programmerclick.com/article/66431189170/
    xr = []
    yr = []

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Catch cordinates of point
            xr.append(x)
            yr.append(y)
            # Draw point in screen
            cv2.circle(img, (x, y), 3, (255, 0, 0), thickness = -1)
            cv2.imshow("Imagen", img)

    cv2.namedWindow("Imagen")
    cv2.setMouseCallback("Imagen", on_EVENT_LBUTTONDOWN)

    while True:
        cv2.imshow("Imagen", img)
        # Wait space key (ASCII 32) to end proccess
        if ( cv2.waitKey(0) & 0xFF == 32 ):
            break

    cv2.destroyAllWindows()

    return xr, yr
    # return ([xr, yr])

import cv2 as cv2

count = 0
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
            cv2.imshow("Image", img)
            global count
            count += 1
            print('count = {}'.format(count))

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", on_EVENT_LBUTTONDOWN)

    while True:
        cv2.imshow("Image", img)
        # Wait space key (ASCII 32) to end proccess
        if ( cv2.waitKey(0) & 0xFF == 32 ):
            break

    cv2.destroyAllWindows()
    return xr, yr
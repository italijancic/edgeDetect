import cv2
import numpy as np

#Podría plantear los puntos graficados con matplotlib (plt) scatter() y actualizar los mismos...

img = cv2.imread('Prueba.jpg') #Almaceno la img original
height, width = img.shape[:2] #Guardo las dimensiones de la img

#https://www.iteramos.com/pregunta/18906/encontrar-el-valor-mas-cercano-en-el-array-de-numpy
def Find_Nearest1(array, value): 
    idx = np.array([np.linalg.norm(x + y) for (x, y) in abs(array - value)]).argmin() 
    return array[idx] 

#https://programmerclick.com/article/66431189170/
def Select_Points1(img):
     
    p_0 = np.random.randint(0, height, (3, 2))
    xr=[]; yr=[]

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            #ind = Find_Nearest1(p_0, [x, y])
            cv2.circle(img, (p_0[0,0], p_0[0,1]), 3, (0, 0, 255), thickness = -1)
            cv2.circle(img, (p_0[1,0], p_0[1,1]), 3, (0, 0, 255), thickness = -1)
            cv2.circle(img, (p_0[2,0], p_0[2,1]), 3, (0, 0, 255), thickness = -1)
        elif event == cv2.EVENT_LBUTTONUP:
            xy = "%d,%d" % (x, y)
            xr.append(x) #Agregado por mi xD
            yr.append(y)
            #cv2.circle(img, (x, y), 3, (255, 0, 0), thickness = -1)
            #cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0,0,0), thickness = 1)
            cv2.imshow("image", img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

    while True:    
        cv2.imshow("image", img)
        if cv2.waitKey(0)&0xFF==32: #Al apretar Espacio (32) se cierra la img
            break
    cv2.destroyAllWindows()

    return([xr,yr])

Select_Points1(img)

# https://docs.opencv.org/3.4/db/d5b/tutorial_py_mouse_handling.html
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
cv2.destroyAllWindows() 
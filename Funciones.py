import numpy as np
import cv2 as cv2
import os
import matplotlib.pyplot as plt

def Find_Nearest(array, value): 
    #https://www.iteramos.com/pregunta/18906/encontrar-el-valor-mas-cercano-en-el-array-de-numpy
    idx = np.array([np.linalg.norm(x + y) for (x, y) in abs(array - value)]).argmin() 
    return array[idx] 

def Select_Points(img):
    #https://programmerclick.com/article/66431189170/
    xr=[]; yr=[]

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xr.append(x) #Agregado
            yr.append(y) #Agregado 
            cv2.circle(img, (x, y), 3, (100, 100, 100), thickness = -1)
            cv2.imshow("Imagen", img)

    cv2.namedWindow("Imagen")
    cv2.setMouseCallback("Imagen", on_EVENT_LBUTTONDOWN)

    while True:    
        cv2.imshow("Imagen", img)
        if cv2.waitKey(0)&0xFF==32: #Al apretar Espacio (32) se cierra la img
            break
    cv2.destroyAllWindows()

    return([xr, yr])

def Pointsv1(data): #Código similar a como lo pensé en el Mathematica
    #Diferencia importante en velocidad entre np.append() y list.append(): https://towardsdatascience.com/python-lists-are-sometimes-much-faster-than-numpy-heres-a-proof-4b3dad4653ad
    dim = np.shape(data)
    puntos = []

    for i in range(0, dim[0], 1):
        for j in range(0, dim[1], 1):
            if data[i, j] == 0:
                puntos.append([j, i])
                break

    for i in range(0, dim[0], 1):
        for j in range(dim[1]-1, 0, -1):
            if data[i,j] == 0:
                puntos.append([j, i])
                break   

    for j in range(dim[1]-1, 0, -1):
        for i in range(0, dim[0], 1):
            if data[i, j] == 0:
                puntos.append([j, i])
                break

    for j in range(dim[1]-1, 0, -1):
        for i in range(dim[0]-1, 0, -1):
            if data[i,j] != 0:
                puntos.append([j, i])
                break

    puntos = np.unique(puntos, axis=0) #https://programmerclick.com/article/9304717476/
    puntos = np.array(puntos)

    return(puntos)

def Pointsv2(edges):
    indexs = np.asarray(np.where(edges != [0]))
    puntos = np.array(list(zip(indexs[0], indexs[1])))
    return(puntos)

def Solve_Circle(x1, y1, x2, y2, x3, y3) :
    #https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/

    x12 = x1 - x2
    x13 = x1 - x3
 
    y12 = y1 - y2
    y13 = y1 - y3
 
    y31 = y3 - y1
    y21 = y2 - y1
 
    x31 = x3 - x1
    x21 = x2 - x1
 
    sx13 = x1 ** 2 - x3 ** 2
    sy13 = y1 ** 2 - y3 ** 2
    sx21 = x2 ** 2 - x1 ** 2
    sy21 = y2 ** 2 - y1 ** 2
 
    f = (sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) // (2 * (y31 * x12 - y21 * x13))          
    g = (sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) // (2 * (x31 * y12 - x21 * y13))
    c = -(x1 ** 2) - y1 ** 2 - 2 * g * x1 - 2 * f * y1
 
    #Ecuación del círculo como: x ^ 2 + y ^ 2 + 2 * g * x + 2 * f * y + c = 0
    h = -g #El centro esta en (h = -g, k = -f)
    k = -f     
    r = round(np.sqrt(h ** 2 + k ** 2 - c), 5) #El radio es: r^2 = h^2 + k^2 - c

    return [h,k,r]


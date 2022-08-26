import cv2
import numpy as np

def getEdgePoints(edges):
    edgePoints = np.asarray(np.where(edges != [0]))
    puntos = np.array(list(zip(edgePoints[1], edgePoints[0])))
    return puntos

def findNearest(array, value):
    #https://www.iteramos.com/pregunta/18906/encontrar-el-valor-mas-cercano-en-el-array-de-numpy
    idx = np.array([np.linalg.norm(x + y) for (x, y) in abs(array - value)]).argmin()
    return array[idx]




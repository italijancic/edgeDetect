import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Funciones import *

#https://www.codespeedy.com/convert-rgb-to-binary-image-in-python/

# Load img from file
img = cv2.imread('img/test.jpeg', 0)
# img = cv2.imread('img/Prueba.jpg', 0)
# Save img size (height, width)
height, width = img.shape[:2]
# Binarize img
ret, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Detect edge
#edges = np.asarray(img_bin) #Transformo la img_bin en un array
edges = cv2.Canny(img, 127, 255)

# Get edge points
edgePoints = getEdgePoints(edges)
print("edgePoints = ", edgePoints)

plt.figure()
plt.plot()
plt.title('cv2.Canny')
plt.imshow(img)
plt.plot(edgePoints[1,:], edgePoints[0,:], '.w')
plt.show(block = False)

# El ussuario selecciona los puntos de la img: los primeros dos definen la recta y el tercero el círculo
xr, yr = getSelectedPoints(img)[:2]

# print("Selected Points (x, y): ", xr[0], yr[0],xr[1], yr[1], xr[2], yr[2])
f'({xr[0]}, {yr[0]}), ({xr[1]}, {yr[1]}),({xr[2]}, {yr[2]})'

# Fit circle, using user selected points
data_cir = fitCircle(xr[0], yr[0],xr[1], yr[1], xr[2], yr[2])
# Here we can use other fit function
xc, yc, r = data_cir[:3]

# Fit line to user selected points
m = (yr[1] - yr[0]) / (xr[1] - xr[0])
b = yr[0] - m * xr[0]

# Calculate numerical values of estimated line
x_axis = np.linspace(0, width, width)
y_axis = m * x_axis + b

#Filtrado de puntos pertenecientes a la gota en sí:
# puntos_aux = []
# for i in range(0,len(edgePoints), 1):
#     if m * edgePoints[i,0] + b >= edgePoints[i,1]: #De acuerdo a la recta trazada
#         if pow((edgePoints[i,0] - xc) ** 2 + (edgePoints[i,1] - yc) ** 2, 1/2) >= r - 5: #De acuerdo al círculo menor
#             if pow((edgePoints[i,0] - xc) ** 2 + (edgePoints[i,1] - yc) ** 2, 1/2) <= r + 5: #De acuerdo al círculo mayor
#                 puntos_aux. append(edgePoints[i]) #El (0,0) esta en la esquina superior derecha de la img

# puntos_aux = np.array(puntos_aux)
#puntos_aux = np.unique(puntos_aux, axis = 0)

# x = puntos_aux[:,0]
# y = puntos_aux[:,1]
# print("Aprox points","x = ", x, "y = ", y)

# x = []; y = [] #Separo las coordenadas de los puntos en x e y
# for i in range(len(puntos_aux)):
#     x.append(puntos_aux[i,0])
#     y.append(puntos_aux[i,1])


#Hallo los puntos más cercanos a los elegidos por el usuario que pertenecen a los puntos filtrados
p1 = [xr[0], yr[0]]
p2 = [xr[1], yr[1]]
p3 = [xr[2], yr[2]]
p4 = [xr[3], yr[3]]
p5 = [xr[4], yr[4]]
# p4 = [xr[0], yr[0] + 2/3 * (yr[2] - yr[0])]
# p5 = [xr[1], yr[1] + 2/3 * (yr[2] - yr[1])]

# p1 = findNearest(puntos_aux, [xr[0], yr[0]])
# p2 = findNearest(puntos_aux, [xr[1], yr[1]])
# p3 = findNearest(puntos_aux, [xr[2], yr[2]])
# p4 = findNearest(puntos_aux, [xr[0], yr[0] + 2/3 * (yr[2] - yr[0])])
# p5 = findNearest(puntos_aux, [xr[1], yr[1] + 2/3 * (yr[2] - yr[1])])
#Ajusto una elipse rotada
e = cv2.fitEllipseDirect(np.asarray([p1, p2, p3, p4, p5]))
# e = cv2.fitEllipseDirect(np.asarray([p1, p2, p3, p4, p5]))

def Coefficients(e):
    #https://stackoverflow.com/questions/32793703/how-can-i-get-ellipse-coefficient-from-fitellipse-function-of-opencv
    #Ellipse(Point(xc, yc), Size(a, b), theta)
    xc = e[0][0]; yc = e[0][1]
    a = e[1][0] / 2; b = e[1][1] / 2
    theta = np.radians(e[2])

    # https://en.wikipedia.org/wiki/Ellipse : Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 is the equation
    A = a * a * pow(np.sin(theta), 2) + b * b * pow(np.cos(theta), 2)
    B = 2 * (b * b - a * a) * np.sin(theta) * np.cos(theta)
    C = a * a * pow(np.cos(theta), 2) + b * b * pow(np.sin(theta), 2)
    D = - 2 * A * xc - B * yc
    E = - B * xc - 2 * C * yc
    F = A * xc * xc + B * xc * yc + C * yc * yc - a * a * b * b
    coef = np.array([A, B, C, D, E, F]) / F

    return coef

# A, B, C, D, E, F = Coefficients(e)[:6]
A, B, C, D, E, F = Coefficients(e)[:6]

print('a, b, c, d, e, f = ', A, B, C, D, E, F)

# Polinomio solución deducido en el Mathematica de reemplazar en la ecuación de la cónica "y" por m*x+b, ecuación de segundo grado
coef_pol_0 = b * b * C + b * E + F
coef_pol_1 = b * B + D + 2 * b * C * m + E * m
coef_pol_2 = A + B * m + C * m * m

print("coef_pol_0", coef_pol_0)
print("coef_pol_1", coef_pol_1)
print("coef_pol_2", coef_pol_2)

# Intersección cónica - recta a partir del polinomio solución
# p = np.polynomial.polynomial.Polynomial((coef_pol_0, coef_pol_1, coef_pol_2), domain = [0, width], window = [0, width])
p = np.polynomial.Polynomial((coef_pol_0, coef_pol_1, coef_pol_2), domain = [0, width], window = [0, width])
raices = p.roots()
raices_xy = np.array([[raices[0], m * raices[0] + b], [raices[1], m * raices[1] + b]])

# Derivada en el punto de forma implicita
der_1 = (- D - 2 * A * raices_xy[0, 0] - B * raices_xy[0, 1]) / (E + 2 * B * raices_xy[0, 0] + 2 * C * raices_xy[0, 1])
der_2 = (- D - 2 * A * raices_xy[1, 0] - B * raices_xy[1, 1]) / (E + 2 * B * raices_xy[1, 0] + 2 * C * raices_xy[1, 1])

der_1_str = str(np.arctan(der_1) * 180 / np.pi)
der_2_str = str(np.arctan(der_2) * 180 / np.pi)

print("La derivada es: " + der_1_str)
print("La derivada es: " + der_2_str)

m1 = np.arctan(der_1)
m2 = np.arctan(der_2)

y_axis_1 = m1 * (x_axis - raices_xy[0, 0]) + raices_xy[0, 1]
y_axis_2 = m2 * (x_axis - raices_xy[1, 0]) + raices_xy[1, 1]

# Círculo, debe ir primero para graficar
cv2.circle(img, (xc, yc) , int(r), (255, 0, 0), thickness = 1)
# Elipse, debe ir primero para graficar
cv2.ellipse(img, e, (255,0,0), thickness = 1)

# Matplotlib figure
plt.figure()
# Imagen original
plt.imshow(img)
# Puntos del contorno ya filtrados
# plt.scatter(x,y)
# Puntos seleccionados
plt.scatter(xr,yr)
#Recta
plt.plot(x_axis, y_axis)
#Recta
plt.plot(x_axis, y_axis_1)
# Recta
plt.plot(x_axis, y_axis_2)


if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()
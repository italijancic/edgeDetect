import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ours modules
from catchPoints import getSelectedPoints
from cv2Elipse import fitElipse
from edgeProccess import findNearest, getEdgePoints
from circleFit import fitCircle
from lineFit import twoDotsLineFit
from filtersPoints import filterSelectedPoints

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

plt.figure()
plt.plot()
plt.title('cv2.Canny')
plt.imshow(img)
plt.plot(edgePoints[:,0], edgePoints[:,1], '.w')
# plt.plot(edgePoints[:,1], edgePoints[:,0], '.w')
plt.show(block = False)

# El ussuario selecciona los puntos de la img: los primeros dos definen la recta y el tercero el círculo
xr, yr = getSelectedPoints(img)
print('\r\nSelected Points: ({}, {}), ({}, {}),({}, {})'.format(xr[0], yr[0],xr[1], yr[1], xr[2], yr[2]))

# Fit circle, using user selected points
data_cir = fitCircle(xr[0], yr[0],xr[1], yr[1], xr[2], yr[2])
# Here we can use other fit function
xc, yc, r = data_cir

# Horizontal line fit
m, b = twoDotsLineFit(xr, yr)
print('\r\nAproximate horizontal line')
print('y = {}x + {}'.format(m, b))
# Calculate numerical values of estimated line
xh = np.linspace(0, width, width)
yh = m * xh + b

# Filter user selected points
puntos_aux = filterSelectedPoints(edgePoints, xc, yc, r, m, b)

# Hallo los puntos más cercanos a los elegidos por el usuario que pertenecen a los puntos filtrados
p1 = findNearest(puntos_aux, [xr[0], yr[0]])
p2 = findNearest(puntos_aux, [xr[1], yr[1]])
p3 = findNearest(puntos_aux, [xr[2], yr[2]])
p4 = findNearest(puntos_aux, [xr[0], yr[0] + 2/3 * (yr[2] - yr[0])])
p5 = findNearest(puntos_aux, [xr[1], yr[1] + 2/3 * (yr[2] - yr[1])])

print('\r\nNearest points')
print('({}), ({}), ({}), ({}), ({})'.format(p1,p2,p3,p4,p5))
e, coefs = fitElipse(p1, p2, p3, p4, p5)
A, B, C, D, E, F = coefs
print('a, b, c, d, e, f = ', A, B, C, D, E, F)

# Polinomio solución deducido en el Mathematica de reemplazar en la ecuación de la cónica "y" por m*x+b, ecuación de segundo grado
coef_pol_0 = b * b * C + b * E + F
coef_pol_1 = b * B + D + 2 * b * C * m + E * m
coef_pol_2 = A + B * m + C * m * m

print("\r\ncoef_pol_0", coef_pol_0)
print("coef_pol_1", coef_pol_1)
print("coef_pol_2", coef_pol_2)

# Intersección cónica - recta a partir del polinomio solución
# p = np.polynomial.Polynomial((coef_pol_0, coef_pol_1, coef_pol_2), domain = [0, width], window = [0, width])
# raicesOld = p.roots()
raices = np.roots([coef_pol_2, coef_pol_1, coef_pol_0])
print('\r\nraices = {}'.format(raices))
# print('rootsOld = {}'.format(raicesOld))

# Calculate y coordenate for roots
raices_xy = np.array([[raices[0], m * raices[0] + b], [raices[1], m * raices[1] + b]])

print('\r\nraices_xy = ', raices_xy)

# Derivada en el punto de forma implicita
m1 = (- D - 2 * A * raices_xy[0, 0] - B * raices_xy[0, 1]) / (E +  B * raices_xy[0, 0] + 2 * C * raices_xy[0, 1])
m2 = (- D - 2 * A * raices_xy[1, 0] - B * raices_xy[1, 1]) / (E +  B * raices_xy[1, 0] + 2 * C * raices_xy[1, 1])

# Pass from pendt to angle
der_1_str = str(np.arctan(m1) * 180 / np.pi)
der_2_str = str(np.arctan(m2) * 180 / np.pi)
print("ThetaC: ", der_1_str, "[degree]")
print("ThetaC: ", der_2_str, "[degree]")

ytg1 = m1 * (xh - raices_xy[0, 0]) + raices_xy[0, 1]
ytg2 = m2 * (xh - raices_xy[1, 0]) + raices_xy[1, 1]

# Círculo, debe ir primero para graficar
cv2.circle(img, (xc, yc) , int(r), (255, 0, 0), thickness = 1)
# Elipse, debe ir primero para graficar
cv2.ellipse(img, e, (255,0,0), thickness = 1)
# Matplotlib figure
plt.figure()
# Puntos del contorno ya filtrados
# plt.scatter(filterX,filterY)
# Puntos seleccionados
plt.scatter(xr,yr)
#Recta
plt.plot(xh, yh)
# Recta tg 1
plt.plot(xh, ytg1)
# Recta tg 2
plt.plot(xh, ytg2)

# Neares Points
plt.plot(p1[0], p1[1], '*r')
plt.plot(p2[0], p2[1], '*r')
plt.plot(p3[0], p3[1], '*r')
plt.plot(p4[0], p4[1], '*r')
plt.plot(p5[0], p5[1], '*r')

# Imagen original
plt.imshow(img)

if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()
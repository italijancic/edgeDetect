import cv2
import numpy as np

#https://stackoverflow.com/questions/32793703/how-can-i-get-ellipse-coefficient-from-fitellipse-function-of-opencv
def fitElipse(p1, p2, p3, p4, p5):
    # Ajusto una elipse rotada
    e = cv2.fitEllipseDirect(np.asarray([p1, p2, p3, p4, p5]))

    #Ellipse(Point(xc, yc), Size(a, b), theta)
    xc = e[0][0]
    yc = e[0][1]
    a = e[1][0] / 2
    b = e[1][1] / 2
    theta = np.radians(e[2])

    # https://en.wikipedia.org/wiki/Ellipse : Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 is the equation
    A = a * a * pow(np.sin(theta), 2) + b * b * pow(np.cos(theta), 2)
    B = 2 * (b * b - a * a) * np.sin(theta) * np.cos(theta)
    C = a * a * pow(np.cos(theta), 2) + b * b * pow(np.sin(theta), 2)
    D = - 2 * A * xc - B * yc
    E = - B * xc - 2 * C * yc
    F = A * xc * xc + B * xc * yc + C * yc * yc - a * a * b * b
    coef = np.array([A, B, C, D, E, F]) / F

    return e, coef

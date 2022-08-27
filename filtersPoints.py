import numpy as np

def filterSelectedPoints(edgePoints, xc, yc, r, m, b):
	#Filtrado de puntos pertenecientes a la gota en sí:
	puntos_aux = []
	for i in range(0,len(edgePoints), 1):
		if m * edgePoints[i,0] + b >= edgePoints[i,1]: #De acuerdo a la recta trazada
			if pow((edgePoints[i,0] - xc) ** 2 + (edgePoints[i,1] - yc) ** 2, 1/2) >= r - 5: #De acuerdo al círculo menor
				if pow((edgePoints[i,0] - xc) ** 2 + (edgePoints[i,1] - yc) ** 2, 1/2) <= r + 5: #De acuerdo al círculo mayor
					puntos_aux.append(edgePoints[i]) #El (0,0) esta en la esquina superior derecha de la img

	puntos_aux = np.array(puntos_aux)
	return puntos_aux
	#puntos_aux = np.unique(puntos_aux, axis = 0)
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

img_arr = mpimg.imread("Prueba.jpg")

def cargar_img():
    path_image = filedialog.askopenfilename(filetypes = [("image", ".jpeg"), ("image", ".png"), ("image", ".jpg")])
    img_arr = mpimg.imread(path_image)
    plt.imshow(img_arr)
    plt.show()

#Otro Ejemplo https://stackoverflow.com/questions/47517718/embedding-image-in-matplotlib-and-displaying-it-in-tkinter
root = tk.Tk()
root.geometry('720x480')
root.wm_title('Medición de Ángulo de Contacto')
root.minsize(width = 720, height = 480)

# Creamos el botón para elegir la imagen de entrada
btn = tk.Button(root, text="Elegir imagen", width=25, command = cargar_img)
btn.pack(side = "top")

f = Figure()
a = f.add_subplot(111)
a.imshow(img_arr)

canvas = FigureCanvasTkAgg(f, master = root)
canvas.draw()
canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
canvas._tkcanvas.pack(side="top", fill="both", expand=1)

root.mainloop()
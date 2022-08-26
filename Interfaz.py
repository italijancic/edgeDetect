import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from catchPoints import *

def open_file():
    browse_text.set("Esperando...")
    file = filedialog.askopenfile(parent=root, mode='r', title="Selección de imagen", filetypes = [("Imagen", ".jpeg, .jpg")])
    if file:
        img = plt.imread(file.name) #https://stackoverflow.com/questions/24564889/opencv-python-not-opening-images-with-imread/60749818#60749818
        if img is None:
            messagebox.showerror(title="Ya casi", message= "Sigue intentando")
            browse_text.set("Cargar imagen")
        else:
            getSelectedPoints(img)
            browse_text.set("Cargar imagen")

#Ventana
root = tk.Tk()
root.title("Medición Ángulo de Contacto")
root.geometry("480x360")
root.maxsize(1980, 1080)
root.minsize(360, 240)

#Figura sobre la que dibujo
canvas = tk.Canvas(root, width=360, height=240)
canvas.place(relx= 0, rely= 0, anchor= "center")

#Instrucciones
instructions = tk.Label(root, text = "Cargue la imagen y seleccione los tres puntos \n que delimitan el contorno de la gota.", font = ("Arial", 15))
instructions.place(relx= 0.5, rely= 0.2, anchor= "center")

#Boton
browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable= browse_text, command= open_file, font="Arial", bg="#2823da", fg="white", height=2, width=15)
browse_text.set("Cargar Imagen")
browse_btn.place(relx= 0.5, rely= 0.6, anchor= "center")

root.mainloop()
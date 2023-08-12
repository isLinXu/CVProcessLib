import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps, ImageFilter
import cv2
import numpy as np

class ImageProcessingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Image Processing GUI")

        self.original_image = None
        self.processed_image = None

        self.open_button = tk.Button(master, text="Open Image", command=self.open_image)
        self.open_button.pack()

        self.save_button = tk.Button(master, text="Save Image", command=self.save_image)
        self.save_button.pack()

        self.canvas = tk.Canvas(master, width=800, height=400)
        self.canvas.pack()

        self.gray_button = tk.Button(master, text="Grayscale", command=self.convert_grayscale)
        self.gray_button.pack()

        self.edge_button = tk.Button(master, text="Edge Detection", command=self.edge_detection)
        self.edge_button.pack()

        self.threshold_label = tk.Label(master, text="Threshold")
        self.threshold_label.pack()

        self.threshold_slider = tk.Scale(master, from_=0, to=255, orient=tk.HORIZONTAL)
        self.threshold_slider.pack()

        self.threshold_button = tk.Button(master, text="Apply Threshold", command=self.apply_threshold)
        self.threshold_button.pack()

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = Image.open(file_path)
            self.processed_image = self.original_image.copy()
            self.show_images()

    def save_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                self.processed_image.save(file_path)

    def show_images(self):
        if self.original_image and self.processed_image:
            original_image = self.original_image.resize((400, 400), Image.ANTIALIAS)
            processed_image = self.processed_image.resize((400, 400), Image.ANTIALIAS)

            original_photo = ImageTk.PhotoImage(original_image)
            processed_photo = ImageTk.PhotoImage(processed_image)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=original_photo)
            self.canvas.create_image(400, 0, anchor=tk.NW, image=processed_photo)

            self.canvas.image1 = original_photo
            self.canvas.image2 = processed_photo

    def convert_grayscale(self):
        if self.original_image:
            self.processed_image = ImageOps.grayscale(self.original_image)
            self.show_images()

    def edge_detection(self):
        if self.original_image:
            img = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            edges = cv2.Canny(img, 100, 200)
            self.processed_image = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
            self.show_images()

    def apply_threshold(self):
        if self.original_image:
            threshold = self.threshold_slider.get()
            img = self.original_image.convert("L")
            self.processed_image = img.point(lambda p: p > threshold and 255)
            self.show_images()

if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageProcessingGUI(root)
    root.mainloop()
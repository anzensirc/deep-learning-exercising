import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# === Load Model ===
model = tf.keras.models.load_model("sports_balls_classifier.h5")

# Class Names (harus sama urutan dengan training)
class_names = ['american_football', 'baseball', 'basketball', 'billiard_ball', 'bowling_ball', 'cricket_ball', 'football', 'golf_ball', 'hockey_ball', 'hockey_puck', 'rugby_ball','shuttlecock', 'table_tennis_ball', 'volleyball']  # ganti sesuai dataset

IMG_SIZE = (224, 224)

def predict_image(file_path):
    img = tf.keras.utils.load_img(file_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0) / 255.0

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_id = np.argmax(score)
    return class_names[class_id], 100 * np.max(score)

def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
    )
    if not file_path:
        return

    # Tampilkan gambar di GUI
    img = Image.open(file_path)
    img.thumbnail((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    label_img.config(image=img_tk)
    label_img.image = img_tk

    # Prediksi
    predicted_class, confidence = predict_image(file_path)
    label_result.config(
        text=f"Prediksi: {predicted_class}\nAkurasi: {confidence:.2f}%", fg="blue"
    )

# === Setup GUI ===
root = tk.Tk()
root.title("Klasifikasi Jenis Bola")
root.geometry("400x500")

btn_open = Button(root, text="Pilih Gambar", command=open_image)
btn_open.pack(pady=10)

label_img = Label(root)
label_img.pack(pady=10)

label_result = Label(root, text="Belum ada gambar", font=("Arial", 14))
label_result.pack(pady=20)

root.mainloop()

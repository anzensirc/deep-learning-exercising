import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# =========================
# Konfigurasi & Tema
# =========================
THEME = {
    "bg":        "#0f172a",  # slate-900
    "panel":     "#111827",  # gray-900
    "card":      "#0b1220",  # trough/bg elemen
    "text":      "#e5e7eb",  # gray-200
    "muted":     "#94a3b8",  # slate-400
    "accent":    "#22c55e",  # green-500
    "accent2":   "#16a34a",  # green-600
    "border":    "#1f2937",  # gray-800
    "warning":   "#f59e0b",  # amber
    "danger":    "#ef4444",  # red
}

FONT_H1   = ("Segoe UI", 20, "bold")
FONT_H2   = ("Segoe UI", 14, "bold")
FONT_BASE = ("Segoe UI", 11)
FONT_MUTE = ("Segoe UI", 10)
FONT_PCT  = ("Segoe UI", 18, "bold")

IMG_SIZE = (224, 224)  # untuk preprocessing model
PREVIEW_MAX = (360, 360)  # untuk tampilan di UI

class_names = [
    'american_football', 'baseball', 'basketball', 'billiard_ball',
    'bowling_ball', 'cricket_ball', 'football', 'golf_ball',
    'hockey_ball', 'hockey_puck', 'rugby_ball', 'shuttlecock',
    'table_tennis_ball', 'volleyball'
]

ICON = {
    'basketball': '🏀', 'football': '⚽', 'american_football': '🏈',
    'volleyball': '🏐', 'table_tennis_ball': '🏓', 'golf_ball': '⛳',
    'billiard_ball': '🎱', 'baseball': '⚾', 'hockey_puck': '🏒',
    'hockey_ball': '🏑', 'cricket_ball': '🏏', 'rugby_ball': '🏉',
    'shuttlecock': '🏸', 'bowling_ball': '🎳'
}
DEFAULT_ICON = "🏅"

# =========================
# Load Model (aman dengan fallback)
# =========================
MODEL_PATH = "sports_balls_classifier.h5"
model = None
model_err = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model '{MODEL_PATH}' tidak ditemukan.")
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model_err = str(e)

# =========================
# Fungsi Util
# =========================
def predict_top3(file_path):
    if model is None:
        raise RuntimeError(model_err or "Model belum siap.")
    img = tf.keras.utils.load_img(file_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)[None] / 255.0
    preds = tf.nn.softmax(model.predict(img_array, verbose=0)[0]).numpy()

    top3_idx = preds.argsort()[-3:][::-1]
    results = []
    for i in top3_idx:
        if i < len(class_names):
            results.append((class_names[i], float(preds[i] * 100)))
    return results

def shorten_label(name: str) -> str:
    # Sedikit rapihkan penamaan untuk tampilan
    return name.replace("_", " ").title()

def icon_for(name: str) -> str:
    return ICON.get(name, DEFAULT_ICON)

# =========================
# UI Komponen
# =========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Klasifikasi Jenis Bola — Modern UI")
        self.configure(bg=THEME["bg"])
        self.geometry("880x580")
        self.minsize(780, 520)

        # Style ttk
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=THEME["bg"])
        style.configure("TLabel", background=THEME["bg"], foreground=THEME["text"], font=FONT_BASE)
        style.configure("Muted.TLabel", foreground=THEME["muted"], background=THEME["bg"], font=FONT_MUTE)
        style.configure("TButton", background=THEME["accent"], foreground="#0b0f19", font=FONT_BASE, borderwidth=0, focusthickness=3)
        style.map("TButton", background=[("active", THEME["accent2"])])

        # Header
        header = tk.Frame(self, bg=THEME["bg"])
        header.pack(fill="x", padx=16, pady=(16, 8))

        tk.Label(header, text="Klasifikasi Jenis Bola", font=FONT_H1, fg=THEME["text"], bg=THEME["bg"]).pack(anchor="w")
        tk.Label(header, text="Pilih gambar → lihat Top-3 prediksi dengan persentase yang jelas.",
                 font=FONT_MUTE, fg=THEME["muted"], bg=THEME["bg"]).pack(anchor="w", pady=(2, 0))

        # Toolbar
        toolbar = tk.Frame(self, bg=THEME["bg"])
        toolbar.pack(fill="x", padx=16, pady=(0, 8))

        self.btn_choose = ttk.Button(toolbar, text="Pilih Gambar", command=self.open_image)
        self.btn_choose.pack(side="left")

        # Main split
        main = tk.Frame(self, bg=THEME["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=16)

        self.left = Panel(main, title="Pratinjau Gambar")
        self.left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.right = Panel(main, title="Hasil Prediksi")
        self.right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        # Isi panel kiri (placeholder)
        self.preview = ImagePreview(self.left.body, width=PREVIEW_MAX[0], height=PREVIEW_MAX[1])
        self.preview.pack(fill="both", expand=True)

        # Isi panel kanan (state awal)
        self.result_view = ResultView(self.right.body)
        self.result_view.pack(fill="both", expand=True)

        if model is None:
            self.result_view.show_message(
                title="Model belum siap",
                message=model_err or "Model tidak dapat dimuat.",
                kind="danger"
            )
        else:
            self.result_view.show_message(
                title="Belum ada prediksi",
                message="Silakan pilih gambar untuk memulai.",
                kind="muted"
            )

    # === Event ===
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")]
        )
        if not file_path:
            return

        # Tampilkan pratinjau
        self.preview.set_image(file_path)

        # Prediksi & render
        try:
            top3 = predict_top3(file_path)
            self.result_view.render_predictions(top3)
        except Exception as e:
            self.result_view.show_message(
                title="Gagal memproses",
                message=str(e),
                kind="danger"
            )


class Panel(tk.Frame):
    def __init__(self, parent, title=""):
        super().__init__(parent, bg=THEME["bg"])
        # Card wrapper (border halus)
        self.card = tk.Frame(self, bg=THEME["panel"], highlightthickness=1, highlightbackground=THEME["border"])
        self.card.pack(fill="both", expand=True)

        self.header = tk.Frame(self.card, bg=THEME["panel"])
        self.header.pack(fill="x", padx=14, pady=(12, 8))
        tk.Label(self.header, text=title, font=FONT_H2, fg=THEME["text"], bg=THEME["panel"]).pack(anchor="w")

        self.body = tk.Frame(self.card, bg=THEME["panel"])
        self.body.pack(fill="both", expand=True, padx=14, pady=14)


class ImagePreview(tk.Frame):
    def __init__(self, parent, width=360, height=360):
        super().__init__(parent, bg=THEME["panel"])
        self.canvas = tk.Canvas(self, width=width, height=height,
                                bg=THEME["card"], highlightthickness=1, highlightbackground=THEME["border"])
        self.canvas.pack(fill="both", expand=True)
        self.width, self.height = width, height
        self._imgtk = None

        # Placeholder teks
        self.placeholder_id = self.canvas.create_text(
            width // 2, height // 2,
            text="Tidak ada gambar.\nKlik 'Pilih Gambar'.",
            fill=THEME["muted"],
            font=FONT_MUTE,
            justify="center"
        )

    def set_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(PREVIEW_MAX)
            self._imgtk = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            # Centering
            w, h = self._imgtk.width(), self._imgtk.height()
            x = (self.width - w) // 2
            y = (self.height - h) // 2
            # Background panel
            self.canvas.configure(bg=THEME["card"])
            self.canvas.create_rectangle(1, 1, self.width-1, self.height-1, outline=THEME["border"])
            self.canvas.create_image(x, y, anchor="nw", image=self._imgtk)
        except Exception as e:
            # Kembali ke placeholder
            self.canvas.delete("all")
            self.canvas.create_text(
                self.width//2, self.height//2,
                text=f"Gagal memuat gambar:\n{e}",
                fill=THEME["danger"],
                font=FONT_MUTE,
                justify="center"
            )


class ResultView(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=THEME["panel"])
        self.container = tk.Frame(self, bg=THEME["panel"])
        self.container.pack(fill="both", expand=True)

    def clear(self):
        for w in self.container.winfo_children():
            w.destroy()

    def show_message(self, title, message, kind="muted"):
        self.clear()
        color = {"muted": THEME["muted"], "danger": THEME["danger"], "warning": THEME["warning"]}.get(kind, THEME["muted"])
        card = Card(self.container)
        card.pack(fill="x", pady=6)
        tk.Label(card.body, text=title, font=FONT_H2, fg=THEME["text"], bg=THEME["panel"]).pack(anchor="w")
        tk.Label(card.body, text=message, font=FONT_BASE, fg=color, bg=THEME["panel"]).pack(anchor="w", pady=(4, 0))

    def render_predictions(self, top3):
        self.clear()
        if not top3:
            self.show_message("Tidak ada hasil", "Model tidak mengembalikan prediksi.", "warning")
            return

        # Kartu besar untuk Top-1
        top1_name, top1_pct = top3[0]
        top1 = EmphasisPredictionCard(self.container, rank=1, label=top1_name, percent=top1_pct)
        top1.pack(fill="x", pady=(0, 10))

        # Kartu ringkas untuk posisi 2 & 3
        for i, (name, pct) in enumerate(top3[1:], start=2):
            card = CompactPredictionCard(self.container, rank=i, label=name, percent=pct)
            card.pack(fill="x", pady=6)


class Card(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=THEME["panel"])
        self.body = tk.Frame(self, bg=THEME["panel"], highlightthickness=1, highlightbackground=THEME["border"])
        self.body.pack(fill="x", padx=0, pady=0)


class EmphasisPredictionCard(Card):
    def __init__(self, parent, rank, label, percent):
        super().__init__(parent)

        head = tk.Frame(self.body, bg=THEME["panel"])
        head.pack(fill="x", padx=12, pady=(12, 8))

        # Rank pill
        pill = tk.Label(head, text=f"#{rank}", font=FONT_BASE, fg=THEME["bg"], bg=THEME["accent"],
                        padx=8, pady=2)
        pill.pack(side="left")

        # Title
        title = f"{icon_for(label)}  {shorten_label(label)}"
        tk.Label(head, text=title, font=FONT_H2, fg=THEME["text"], bg=THEME["panel"]).pack(side="left", padx=10)

        # Big percentage right aligned
        pct_str = f"{percent:.1f}%"
        tk.Label(head, text=pct_str, font=FONT_PCT, fg=THEME["accent"], bg=THEME["panel"]).pack(side="right")

        # Progress bar (custom, smooth)
        bar = PrettyProgress(self.body, percent=percent, height=14)
        bar.pack(fill="x", padx=12, pady=(0, 14))



class CompactPredictionCard(Card):
    def __init__(self, parent, rank, label, percent):
        super().__init__(parent)
        row = tk.Frame(self.body, bg=THEME["panel"])
        row.pack(fill="x", padx=12, pady=10)

        # Kiri: rank + label
        left = tk.Frame(row, bg=THEME["panel"])
        left.pack(side="left", fill="x", expand=True)

        pill = tk.Label(left, text=f"#{rank}", font=FONT_MUTE, fg=THEME["bg"], bg=THEME["muted"], padx=6, pady=1)
        pill.pack(side="left")

        tk.Label(left, text=f"  {icon_for(label)}  {shorten_label(label)}",
                 font=FONT_BASE, fg=THEME["text"], bg=THEME["panel"]).pack(side="left")

        # Kanan: persen
        right = tk.Frame(row, bg=THEME["panel"])
        right.pack(side="right")

        tk.Label(right, text=f"{percent:.1f}%", font=FONT_BASE, fg=THEME["text"], bg=THEME["panel"]).pack(anchor="e")

        bar = PrettyProgress(self.body, percent=percent, height=10)
        bar.pack(fill="x", padx=12, pady=(0, 12))



class PrettyProgress(tk.Frame):
    """Progress bar kustom: trough gelap + bar aksen + label persen di atasnya."""
    def __init__(self, parent, percent=0.0, width=520, height=12):
        super().__init__(parent, bg=THEME["panel"])
        self.width = width
        self.height = height
        self.percent = max(0.0, min(100.0, float(percent)))

        self.trough = tk.Frame(self, bg=THEME["card"], height=height, highlightthickness=1, highlightbackground=THEME["border"])
        self.trough.pack(fill="x")
        self.trough.pack_propagate(False)

        self.bar = tk.Frame(self.trough, bg=THEME["accent"], height=height)
        self.bar.place(x=0, y=0, width=self._bar_width(), height=height)

        self.label = tk.Label(self.trough, text=f"{self.percent:.1f}%", font=("Segoe UI", 9, "bold"),
                              fg=THEME["bg"], bg=THEME["accent"])
        # label di ujung bar, tetap terbaca (fallback jika bar sempit)
        self._place_label()

    def _bar_width(self):
        return int((self.percent / 100.0) * self.width)

    def _place_label(self):
        bw = self._bar_width()
        if bw < 40:
            # Bar terlalu pendek, letakkan label di luar bar agar tetap terbaca
            self.label.configure(bg=THEME["panel"], fg=THEME["text"])
            self.label.place(x=bw + 6, y=-1)
        else:
            self.label.configure(bg=THEME["accent"], fg=THEME["bg"])
            self.label.place(x=max(4, bw - 40), y=-1)


if __name__ == "__main__":
    App().mainloop()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# === 1. Konfigurasi Dataset ===
train_dir = r"train"  # ganti dengan path folder train Anda
test_dir  = r"test"   # ganti dengan path folder test Anda
img_size = (224, 224)
batch_size = 32

# Load dataset awal
train_ds_raw = keras.utils.image_dataset_from_directory(
    train_dir, image_size=img_size, batch_size=batch_size
)
test_ds_raw = keras.utils.image_dataset_from_directory(
    test_dir, image_size=img_size, batch_size=batch_size
)

# Simpan class_names sebelum map & prefetch
class_names = train_ds_raw.class_names
num_classes = len(class_names)
print("Kelas:", class_names)

# Normalisasi data dan prefetch
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds_raw.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

# === 2. Data Augmentation ===
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# === 3. Model Transfer Learning (MobileNetV2) ===
base_model = keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze layer awal

model = keras.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')  # pakai num_classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === 4. Training Model (Transfer Learning) ===
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15
)

# === 5. Fine Tuning (Optional) ===
base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_ds, validation_data=test_ds, epochs=5)

# === 6. Evaluasi ===
loss, acc = model.evaluate(test_ds)
print(f"Akurasi Model: {acc*100:.2f}%")

# === 7. Simpan Model ===
model.save("sports_balls_classifier.h5")
print("Model tersimpan sebagai sports_balls_classifier.h5")

# === 8. Plot Akurasi ===
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')
plt.show()

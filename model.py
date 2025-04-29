from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('model_xray.h5')
IMG_SIZE = 150

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        return {
            "label": "🟢 Paru-paru Normal",
            "penjelasan": "Tidak ditemukan indikasi infeksi. Struktur paru terlihat jernih dan merata.",
            "saran": "Tetap jaga kesehatan dan hindari paparan asap atau polutan."
        }
    else:
        return {
            "label": "🔴 Terindikasi Pneumonia",
            "penjelasan": "Model mendeteksi pola opasitas putih yang menyebar di area paru, khas infeksi.",
            "saran": "Segera periksa ke dokter paru untuk tes lanjutan seperti kultur dahak atau CT scan."
        }

# Quick Start Guide
**Face Mask Detection - Panduan Cepat Memulai**

## 🚀 Setup Cepat (5 Menit)

### 1. Clone Repository
```bash
git clone https://github.com/[username]/face-mask-detection.git
cd face-mask-detection
```

### 2. Install Dependencies
```bash
# Buat virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

### 3. Download Dataset
```bash
# Install Kaggle API
pip install kaggle

# Setup Kaggle credentials
# 1. Login ke kaggle.com
# 2. Go to Account Settings
# 3. Create New API Token
# 4. Move kaggle.json to ~/.kaggle/ (Linux/Mac) atau C:\Users\<username>\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d omkargurav/face-mask-dataset

# Extract
unzip face-mask-dataset.zip -d data/
```

### 4. Struktur Data
Pastikan struktur folder data seperti ini:
```
data/
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── without_mask/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 5. Training Model (Pilih salah satu)

#### Option A: Training Semua Model (Recommended untuk Research)
```bash
python train_all_models.py
```
⏱️ **Waktu:** ~2-4 jam (tergantung hardware)  
📊 **Output:** Semua model + visualisasi perbandingan lengkap

#### Option B: Training Model Terbaik Only (Quick)
Jika Anda hanya ingin model terbaik (MobileNetV2), edit `train_all_models.py`:
```python
# Ubah baris ini (sekitar line 120):
models_to_train = [build_custom_cnn(), build_mobilenet(), build_vgg16()]

# Menjadi:
models_to_train = [build_mobilenet()]
```
Lalu jalankan:
```bash
python train_all_models.py
```
⏱️ **Waktu:** ~30-60 menit

### 6. Prediksi Gambar
```bash
python predict.py --model mobilenetv2 --image path/to/your/image.jpg --visualize
```

## 🎯 Hasil yang Diharapkan

### Training Output
```
Training Model: MobileNetV2
Epoch 1/5
189/189 [==============================] - 45s 238ms/step - loss: 0.2345 - accuracy: 0.9523 - val_loss: 0.1234 - val_accuracy: 0.9801
...
✓ Model final disimpan di: models/MobileNetV2_final.h5
```

### Evaluation Results
```
============================================================
HASIL AKHIR PERBANDINGAN MODEL
============================================================
         Model   Accuracy  Precision    Recall  F1-Score      Params
   Custom_CNN   0.916600   0.917400  0.916600  0.916500  23900000.0
  MobileNetV2   0.980100   0.980100  0.980100  0.980100   2400000.0
        VGG16   0.966900   0.967700  0.966900  0.966900  17900000.0
============================================================
```

## 📊 File-file yang Dihasilkan

Setelah training, Anda akan mendapatkan:

```
face-mask-detection/
├── models/
│   ├── MobileNetV2_best.h5    ← Model terbaik
│   ├── MobileNetV2_final.h5   ← Model akhir
│   └── ...
├── results/
│   ├── evaluation_report.txt  ← Laporan lengkap
│   ├── accuracy_comparison.png
│   ├── confusion_matrices/
│   │   └── all_models_comparison.png
│   └── training_history/
│       └── training_curves.png
```

## 🔧 Troubleshooting

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow==2.13.0
```

### Error: "Could not find data directory"
Pastikan struktur folder `data/` benar dan berisi subfolder `with_mask/` dan `without_mask/`.

### Error: "Out of Memory" saat training
Kurangi batch size di `train_all_models.py`:
```python
BATCH_SIZE = 16  # atau 8
```

### Training terlalu lambat
- Pastikan menggunakan GPU (jika ada)
- Kurangi jumlah epochs:
```python
EPOCHS = 3  # default: 5
```

### Kaggle API tidak bisa download
1. Pastikan file `kaggle.json` sudah di folder yang benar
2. Set permission (Linux/Mac):
```bash
chmod 600 ~/.kaggle/kaggle.json
```

## ⚡ Tips untuk Training Lebih Cepat

### Gunakan GPU
TensorFlow otomatis menggunakan GPU jika tersedia. Cek dengan:
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

### Google Colab (Free GPU)
Jika tidak punya GPU, gunakan Google Colab:

1. Upload project ke Google Drive
2. Buka Google Colab
3. Mount Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
4. Run training:
```python
%cd /content/drive/MyDrive/face-mask-detection
!python train_all_models.py
```

## 📈 Next Steps

Setelah training selesai:

1. **Evaluasi Detail**
```bash
python evaluate.py --model mobilenetv2 --visualize
```

2. **Test dengan Gambar Sendiri**
```bash
python predict.py --model mobilenetv2 --image your_photo.jpg --visualize
```

3. **Lihat Hasil Lengkap**
- Buka `results/evaluation_report.txt`
- Lihat visualisasi di folder `results/`

4. **Eksplorasi Lebih Lanjut**
- Fine-tune hyperparameters
- Tambah augmentasi data
- Coba model lain (ResNet, EfficientNet, etc)

## 🎓 Resources

- [Makalah Lengkap](docs/Analisis-Perbandingan-Model-Deep-Learning-untuk-Deteksi-Masker-Wajah.pdf)
- [Dataset Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)

## 💡 FAQ

**Q: Berapa lama training memakan waktu?**  
A: Tergantung hardware. Dengan GPU: 30-60 menit. Tanpa GPU: 2-4 jam.

**Q: Model mana yang paling bagus?**  
A: MobileNetV2 - akurasi tertinggi (98.01%) dengan parameter paling sedikit (2.4M).

**Q: Bisa dipakai untuk real-time detection?**  
A: Ya! MobileNetV2 sangat cocok untuk real-time karena ringan dan cepat.

**Q: Bagaimana cara deploy model ini?**  
A: Bisa menggunakan TensorFlow Serving, Flask API, atau convert ke TensorFlow Lite untuk mobile.

---

🎉 **Selamat! Anda siap memulai project Face Mask Detection!**

Jika ada pertanyaan, silakan buat issue di GitHub repository.

# Face Mask Detection
**Analisis Komparatif Model Deep Learning untuk Klasifikasi Gambar**

## 👥 Tim Pengembang
- Arnold Oktafianto
- Cahyo Anggoro Seto
- Wahyu Pratama

## 📋 Deskripsi Project
Project ini merupakan studi komparatif yang membandingkan efektivitas tiga arsitektur deep learning untuk deteksi penggunaan masker wajah:
1. **Custom CNN** - Model yang dibangun dari nol
2. **MobileNetV2** - Transfer Learning model yang ringan
3. **VGG16** - Transfer Learning model yang kompleks

Tujuan utama adalah menentukan model terbaik berdasarkan keseimbangan antara akurasi, presisi, recall, dan efisiensi komputasi.

## 🎯 Hasil Utama
**Pemenang: MobileNetV2** dengan performa terbaik di semua metrik:
- ✅ **Akurasi**: 98.01%
- ✅ **Precision**: 98.01%
- ✅ **Recall**: 98.01%
- ✅ **F1-Score**: 98.01%
- ✅ **Parameter**: 2.4M (paling efisien)

### Perbandingan Model

| Model | Accuracy | Precision | Recall | F1-Score | Parameter |
|-------|----------|-----------|--------|----------|-----------|
| **MobileNetV2** | **98.01%** | **98.01%** | **98.01%** | **98.01%** | **2.4M** |
| VGG16 | 96.69% | 96.77% | 96.69% | 96.69% | 17.9M |
| Custom CNN | 91.66% | 91.74% | 91.66% | 91.65% | 23.9M |

## 📊 Dataset
Dataset yang digunakan berasal dari Kaggle:
- **Sumber**: [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- **Total Dataset**: 7,553 gambar
- **Data Training (80%)**: 6,043 gambar
- **Data Testing (20%)**: 1,510 gambar
- **Kelas**: 
  - `with_mask` - Orang menggunakan masker
  - `without_mask` - Orang tidak menggunakan masker

### Karakteristik Dataset
- ✓ Dataset relatif seimbang antara kedua kelas
- ✓ Variasi data mencakup berbagai sudut, pencahayaan, jenis masker, dan latar belakang
- ✓ Simulasi kondisi dunia nyata

### Cara Download Dataset
```bash
# Install Kaggle API
pip install kaggle

# Setup Kaggle credentials (letakkan kaggle.json di ~/.kaggle/)
# Download dari: https://www.kaggle.com/settings -> Create New API Token

# Download dataset
kaggle datasets download -d omkargurav/face-mask-dataset

# Extract file
unzip face-mask-dataset.zip -d data/
```

## 🚀 Instalasi

### Requirements
- Python 3.8 atau lebih tinggi
- GPU (opsional, untuk training lebih cepat)

### Langkah Instalasi
```bash
# Clone repository
git clone https://github.com/[username]/face-mask-detection.git
cd face-mask-detection

# Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset (lihat instruksi di atas)
```

## 📁 Struktur Folder
```
face-mask-detection/
├── data/
│   ├── with_mask/            # Folder untuk gambar dengan masker
│   ├── without_mask/         # Folder untuk gambar tanpa masker
│   └── dataset_info.txt      # Info dataset
├── models/
│   ├── Custom_CNN_best.h5    # Model Custom CNN (best checkpoint)
│   ├── Custom_CNN_final.h5   # Model Custom CNN (final)
│   ├── MobileNetV2_best.h5   # Model MobileNetV2 (best checkpoint)
│   ├── MobileNetV2_final.h5  # Model MobileNetV2 (final)
│   ├── VGG16_best.h5         # Model VGG16 (best checkpoint)
│   └── VGG16_final.h5        # Model VGG16 (final)
├── results/
│   ├── confusion_matrices/   # Confusion matrix untuk setiap model
│   ├── training_history/     # History training (CSV)
│   ├── *.png                 # Visualisasi hasil (accuracy, metrics, ROC, etc)
│   └── evaluation_report.txt # Laporan evaluasi lengkap
├── docs/
│   └── Analisis-Perbandingan-Model-Deep-Learning-untuk-Deteksi-Masker-Wajah.pdf
├── train_all_models.py       # Script training semua model
├── evaluate.py               # Script evaluasi model
├── predict.py                # Script prediksi gambar
├── setup.py                  # Setup untuk instalasi package
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
└── dataset_info.txt
```

## 💻 Cara Menggunakan

### 1. Training Semua Model Sekaligus

Training ketiga model (Custom CNN, MobileNetV2, VGG16) sekaligus dan membuat visualisasi perbandingan:

```bash
python train_all_models.py
```

**Output yang dihasilkan:**
- Model files di folder `models/`
- Training history di `results/training_history/`
- Confusion matrices di `results/confusion_matrices/`
- Visualisasi perbandingan di `results/`
- Laporan lengkap di `results/evaluation_report.txt`

**Catatan:** Training bisa memakan waktu beberapa jam tergantung hardware. Untuk hasil optimal, ubah `EPOCHS` menjadi 10-20 di file `train_all_models.py`.

### 2. Evaluasi Model

Evaluasi model yang sudah dilatih pada validation set:

```bash
# Evaluasi MobileNetV2
python evaluate.py --model mobilenetv2 --visualize

# Evaluasi VGG16
python evaluate.py --model vgg16 --visualize

# Evaluasi Custom CNN
python evaluate.py --model custom_cnn --visualize
```

**Flag `--visualize`** akan menampilkan confusion matrix dan ROC curve.

### 3. Prediksi Gambar Baru

Prediksi single image menggunakan model yang sudah dilatih:

```bash
# Prediksi dengan MobileNetV2 (recommended)
python predict.py --model mobilenetv2 --image path/to/image.jpg --visualize

# Prediksi dengan VGG16
python predict.py --model vgg16 --image path/to/image.jpg --visualize

# Prediksi dengan Custom CNN
python predict.py --model custom_cnn --image path/to/image.jpg --visualize
```

**Output:**
```
============================================================
HASIL PREDIKSI
============================================================
Model: MOBILENETV2
Gambar: test_image.jpg
Prediksi: WITH_MASK
Confidence: 98.45%
============================================================
```

**Flag `--visualize`** akan menampilkan dan menyimpan gambar dengan hasil prediksi.

## 🔬 Metodologi Penelitian

### Training Configuration
- **Epochs**: 5 epochs untuk semua model
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Kondisi**: Sama untuk semua model untuk perbandingan yang adil

### Observasi Penting
- Model Transfer Learning (MobileNetV2 & VGG16) mencapai akurasi tinggi (>95%) sejak epoch pertama berkat pre-trained weights
- Custom CNN membutuhkan waktu lebih lama untuk konvergen
- MobileNetV2 mencapai akurasi ~98% di epoch akhir
- Custom CNN hanya mencapai ~91% dalam 5 epochs yang sama

## 📈 Hasil Evaluasi

### Confusion Matrix Analysis
- **MobileNetV2**: 30 kesalahan total (paling sedikit)
- **VGG16**: 50 kesalahan total
- **Custom CNN**: 126 kesalahan total (paling banyak)

### Kesimpulan Evaluasi
✅ **Jumlah parameter besar ≠ akurasi lebih baik**
- Custom CNN: 23.9M parameter → Akurasi 91.66%
- MobileNetV2: 2.4M parameter → Akurasi 98.01% (10x lebih efisien!)

✅ **Transfer Learning sangat efektif** untuk dataset ini

✅ **MobileNetV2 ideal untuk deployment real-time** pada CCTV atau perangkat mobile

## 🛠️ Teknologi yang Digunakan
- **Deep Learning Framework**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Model Architectures**: 
  - Custom CNN (from scratch)
  - MobileNetV2 (Transfer Learning)
  - VGG16 (Transfer Learning)

## 📄 Dokumentasi Lengkap
Untuk penjelasan detail tentang metodologi, analisis hasil, confusion matrix, dan kesimpulan penelitian, silakan lihat:
- [Makalah Lengkap (PDF)](docs/ANALISIS%20KOMPARATIF%20ARSITEKTUR%20DEEP%20LEARNING%20UNTUK%20KLASIFIKASI%20CITRA%20MASKER%20WAJAH.pdf)

## 🎓 Kesimpulan
**MobileNetV2** adalah model terbaik untuk tugas deteksi masker wajah berdasarkan:
1. ✅ Akurasi tertinggi (98.01%)
2. ✅ Stabilitas metrik terbaik
3. ✅ Efisiensi komputasi superior (2.4M parameter)
4. ✅ Kesalahan prediksi minimal (30 kesalahan)

Model ini sangat cocok untuk implementasi praktis di:
- 📱 Aplikasi mobile
- 📹 Sistem CCTV real-time
- 🚪 Sistem kontrol akses
- 🏥 Monitoring fasilitas kesehatan

## 📝 Lisensi
MIT License

## 🤝 Kontribusi
Contributions, issues, dan feature requests sangat diterima!

## 📞 Kontak
Untuk pertanyaan atau kolaborasi, silakan hubungi tim pengembang.

---
**Note**: Project ini merupakan hasil penelitian komparatif untuk keperluan akademis dan praktis dalam bidang Computer Vision dan Deep Learning.

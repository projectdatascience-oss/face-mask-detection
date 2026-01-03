# Project Structure

Dokumentasi lengkap struktur project Face Mask Detection.

## 📂 Struktur Lengkap

```
face-mask-detection/
│
├── 📄 README.md                    # Dokumentasi utama project
├── 📄 QUICK_START.md               # Panduan cepat memulai
├── 📄 CONTRIBUTING.md              # Panduan kontribusi
├── 📄 GITHUB_UPLOAD_GUIDE.md       # Panduan upload ke GitHub
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Setup untuk instalasi package
├── 📄 .gitignore                   # File yang tidak di-track Git
├── 📄 dataset_info.txt             # Info dataset Kaggle
│
├── 🐍 train_all_models.py          # Script utama untuk training semua model
├── 🐍 evaluate.py                  # Script untuk evaluasi model
├── 🐍 predict.py                   # Script untuk prediksi gambar
│
├── 📁 data/                        # Dataset (NOT in Git)
│   ├── with_mask/                  # Gambar orang dengan masker
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   ├── without_mask/               # Gambar orang tanpa masker
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   └── .gitkeep                    # Untuk track folder kosong
│
├── 📁 models/                      # Trained models (NOT in Git)
│   ├── Custom_CNN_best.h5          # Custom CNN - best checkpoint
│   ├── Custom_CNN_final.h5         # Custom CNN - final model
│   ├── MobileNetV2_best.h5         # MobileNetV2 - best checkpoint ⭐
│   ├── MobileNetV2_final.h5        # MobileNetV2 - final model
│   ├── VGG16_best.h5               # VGG16 - best checkpoint
│   ├── VGG16_final.h5              # VGG16 - final model
│   └── .gitkeep
│
├── 📁 results/                     # Training & evaluation results
│   ├── 📁 confusion_matrices/      # Confusion matrix visualizations
│   │   ├── all_models_comparison.png
│   │   ├── custom_cnn_confusion_matrix.png
│   │   ├── mobilenetv2_confusion_matrix.png
│   │   ├── vgg16_confusion_matrix.png
│   │   └── .gitkeep
│   │
│   ├── 📁 training_history/        # Training history data
│   │   ├── Custom_CNN_history.csv
│   │   ├── MobileNetV2_history.csv
│   │   ├── VGG16_history.csv
│   │   ├── training_curves.png
│   │   └── .gitkeep
│   │
│   ├── 📊 evaluation_report.txt    # Laporan evaluasi lengkap
│   ├── 📊 evaluation_results.csv   # Hasil dalam format CSV
│   ├── 📈 accuracy_comparison.png  # Perbandingan akurasi
│   ├── 📈 metrics_comparison.png   # Perbandingan precision/recall/F1
│   ├── 📈 params_vs_accuracy.png   # Model size vs accuracy
│   ├── 📈 roc_curve_comparison.png # ROC curves
│   ├── 🖼️ class_distribution.png   # Distribusi kelas dataset
│   ├── 🖼️ sample_images.png        # Contoh gambar dataset
│   └── .gitkeep
│
└── 📁 docs/                        # Dokumentasi
    ├── Analisis-Perbandingan-Model-Deep-Learning-untuk-Deteksi-Masker-Wajah.pdf
    └── (tambahkan dokumentasi lain di sini)
```

## 📝 Penjelasan File Utama

### Python Scripts

#### `train_all_models.py` (Main Script)
- **Purpose**: Training ketiga model (Custom CNN, MobileNetV2, VGG16) sekaligus
- **Input**: Dataset di folder `data/`
- **Output**: 
  - Model files di `models/`
  - Visualisasi di `results/`
  - Laporan di `results/evaluation_report.txt`
- **Usage**: `python train_all_models.py`
- **Waktu**: ~2-4 jam (CPU) atau ~30-60 menit (GPU)

#### `evaluate.py`
- **Purpose**: Evaluasi model yang sudah dilatih
- **Input**: Model file dari `models/`
- **Output**: 
  - Confusion matrix
  - ROC curve
  - Classification report
- **Usage**: 
  ```bash
  python evaluate.py --model mobilenetv2 --visualize
  python evaluate.py --model vgg16 --visualize
  python evaluate.py --model custom_cnn --visualize
  ```

#### `predict.py`
- **Purpose**: Prediksi single image
- **Input**: 
  - Model file dari `models/`
  - Image file
- **Output**: 
  - Predicted label (with_mask / without_mask)
  - Confidence score
  - Visualized image (jika --visualize)
- **Usage**: 
  ```bash
  python predict.py --model mobilenetv2 --image test.jpg --visualize
  ```

### Documentation Files

#### `README.md`
- Dokumentasi utama project
- Overview hasil penelitian
- Perbandingan model
- Cara instalasi dan penggunaan

#### `QUICK_START.md`
- Panduan cepat untuk memulai
- Setup dalam 5 menit
- Troubleshooting common issues

#### `CONTRIBUTING.md`
- Guidelines untuk kontributor
- Code style
- Commit message format
- Pull request process

#### `GITHUB_UPLOAD_GUIDE.md`
- Panduan lengkap upload ke GitHub
- Git commands
- Troubleshooting Git issues

#### `dataset_info.txt`
- Info lengkap tentang dataset
- Cara download dari Kaggle
- Struktur folder yang expected

### Configuration Files

#### `requirements.txt`
```
tensorflow==2.13.0
keras==2.13.1
opencv-python==4.8.1.78
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
kaggle==1.5.16
...
```

#### `.gitignore`
Mencegah file-file berikut masuk ke Git:
- Dataset files (data/)
- Model files (models/*.h5)
- Virtual environment (venv/, env/)
- Python cache (__pycache__/)
- Kaggle credentials (kaggle.json)

#### `setup.py`
- Metadata project
- Dependencies
- Installation configuration

#### `LICENSE`
- MIT License
- Copyright information

## 🔄 Workflow Typical

### 1. Setup Awal
```bash
git clone https://github.com/username/face-mask-detection.git
cd face-mask-detection
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
kaggle datasets download -d omkargurav/face-mask-dataset
unzip face-mask-dataset.zip -d data/
```

### 3. Training
```bash
python train_all_models.py
```

### 4. Evaluasi
```bash
python evaluate.py --model mobilenetv2 --visualize
```

### 5. Prediksi
```bash
python predict.py --model mobilenetv2 --image test.jpg --visualize
```

## 📊 Output Files Generated

### After Training (`train_all_models.py`)

```
models/
├── Custom_CNN_best.h5      (23.9M parameters, 91.66% acc)
├── MobileNetV2_best.h5     (2.4M parameters, 98.01% acc) ⭐
└── VGG16_best.h5           (17.9M parameters, 96.69% acc)

results/
├── evaluation_report.txt
├── evaluation_results.csv
├── accuracy_comparison.png
├── metrics_comparison.png
├── params_vs_accuracy.png
├── roc_curve_comparison.png
├── class_distribution.png
├── sample_images.png
├── confusion_matrices/
│   └── all_models_comparison.png
└── training_history/
    ├── Custom_CNN_history.csv
    ├── MobileNetV2_history.csv
    ├── VGG16_history.csv
    └── training_curves.png
```

### After Evaluation (`evaluate.py`)

```
results/
├── mobilenetv2_evaluation_report.txt
├── mobilenetv2_roc_curve.png
└── confusion_matrices/
    └── mobilenetv2_confusion_matrix.png
```

### After Prediction (`predict.py`)

```
test_predicted.jpg  (gambar original dengan label prediksi)
```

## 🎯 Best Practices

### File Naming Convention
- Python files: `lowercase_with_underscores.py`
- Documentation: `UPPERCASE.md`
- Data files: `descriptive_name.extension`

### Code Organization
```python
# 1. Imports
import tensorflow as tf
from tensorflow.keras import layers

# 2. Constants/Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 3. Functions
def load_data():
    pass

def build_model():
    pass

# 4. Main execution
if __name__ == '__main__':
    main()
```

### Git Commit Messages
```bash
git commit -m "feat: add real-time detection"
git commit -m "fix: resolve memory leak in training"
git commit -m "docs: update README with new results"
git commit -m "refactor: improve code readability"
```

## 🚫 What NOT to Commit

### Large Files (>100MB)
- ❌ Dataset images
- ❌ Model .h5 files
- ❌ Large result visualizations

### Sensitive Files
- ❌ `kaggle.json` (API credentials)
- ❌ `.env` files
- ❌ Personal data

### Generated Files
- ❌ `__pycache__/`
- ❌ `.ipynb_checkpoints/`
- ❌ `*.pyc`

All these are already in `.gitignore`!

## 💡 Tips

1. **Keep models/ empty in Git**: Model files terlalu besar, gunakan `.gitkeep` untuk track folder
2. **Commit results/ selectively**: Hanya commit visualisasi penting, skip yang besar
3. **Update README**: Setiap ada hasil baru, update README dengan metrics terbaru
4. **Version control**: Tag setiap major version: `git tag -a v1.0 -m "Version 1.0"`

---

📚 **Untuk informasi lebih lanjut, lihat:**
- [README.md](README.md) - Overview project
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

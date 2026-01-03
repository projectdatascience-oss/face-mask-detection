# Panduan Upload Project ke GitHub

## Langkah-langkah Detail Upload ke GitHub

### 1. Persiapan Awal

#### Install Git (jika belum)
**Windows:**
- Download dari: https://git-scm.com/download/win
- Install dengan setting default

**Mac:**
```bash
# Menggunakan Homebrew
brew install git
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install git
```

#### Verifikasi Instalasi Git
```bash
git --version
# Seharusnya muncul: git version 2.x.x
```

#### Setup Git Identity
```bash
git config --global user.name "Nama Anda"
git config --global user.email "email@anda.com"
```

### 2. Buat Repository di GitHub

1. **Login ke GitHub** (https://github.com)
2. Klik tombol **"+"** di pojok kanan atas → **"New repository"**
3. Isi form:
   - **Repository name**: `face-mask-detection`
   - **Description**: "Analisis Komparatif Model Deep Learning untuk Deteksi Masker Wajah"
   - **Public** atau **Private**: Pilih sesuai kebutuhan
   - ❌ **JANGAN centang** "Add a README file" (kita sudah punya)
   - ❌ **JANGAN centang** "Add .gitignore" (kita sudah punya)
   - **License**: Pilih MIT atau sesuai kebutuhan
4. Klik **"Create repository"**

### 3. Siapkan Struktur Folder Project

Buat struktur folder seperti ini di komputer Anda:

```
face-mask-detection/
├── src/
│   ├── custom_cnn.py
│   ├── mobilenetv2_model.py
│   ├── vgg16_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data/
│   └── dataset_info.txt
├── docs/
│   └── Analisis-Perbandingan-Model-Deep-Learning-untuk-Deteksi-Masker-Wajah.pdf
├── models/
│   └── .gitkeep
├── results/
│   ├── confusion_matrices/
│   │   └── .gitkeep
│   ├── training_history/
│   │   └── .gitkeep
│   └── evaluation_report.txt
├── .gitignore
├── README.md
├── requirements.txt
└── dataset_info.txt
```

**Catatan**: File `.gitkeep` adalah file kosong untuk memaksa Git melacak folder kosong

### 4. Inisialisasi Git di Project

Buka terminal/command prompt, lalu:

```bash
# Masuk ke folder project
cd path/to/face-mask-detection

# Inisialisasi Git
git init

# Cek status
git status
```

### 5. Tambahkan Remote Repository

Ganti `YOUR_USERNAME` dengan username GitHub Anda:

```bash
git remote add origin https://github.com/YOUR_USERNAME/face-mask-detection.git

# Verifikasi
git remote -v
```

### 6. Stage dan Commit Files

```bash
# Tambahkan semua file ke staging area
git add .

# Atau tambahkan file spesifik
# git add README.md
# git add requirements.txt
# git add .gitignore
# git add docs/
# git add src/

# Cek status (file yang akan di-commit)
git status

# Commit dengan pesan
git commit -m "Initial commit: Face mask detection comparative study"
```

### 7. Push ke GitHub

```bash
# Push ke branch main (atau master)
git branch -M main
git push -u origin main
```

Jika diminta login:
- **Username**: username GitHub Anda
- **Password**: gunakan **Personal Access Token** (bukan password biasa)

#### Cara Buat Personal Access Token (jika diperlukan):
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token
3. Beri nama dan pilih scope `repo`
4. Copy token (simpan baik-baik, hanya muncul sekali!)
5. Gunakan token sebagai password saat push

### 8. Verifikasi Upload

1. Buka browser
2. Kunjungi: `https://github.com/YOUR_USERNAME/face-mask-detection`
3. Pastikan semua file sudah terupload

## Update Project (Setelah Initial Upload)

Jika ada perubahan atau file baru:

```bash
# Cek perubahan
git status

# Stage perubahan
git add .
# atau file spesifik: git add src/new_file.py

# Commit
git commit -m "Deskripsi perubahan"

# Push
git push
```

## Tips Penting

### ✅ Yang HARUS di-upload:
- ✅ README.md
- ✅ requirements.txt
- ✅ .gitignore
- ✅ dataset_info.txt
- ✅ File .py (source code)
- ✅ PDF makalah (di folder docs/)
- ✅ File konfigurasi

### ❌ Yang JANGAN di-upload:
- ❌ Dataset (train/ dan test/ folders) → TERLALU BESAR
- ❌ Model .h5 files → TERLALU BESAR
- ❌ Virtual environment (venv/ atau env/)
- ❌ __pycache__/ folder
- ❌ .ipynb_checkpoints/
- ❌ kaggle.json (credentials!)

### Menghapus File yang Salah di-upload

Jika tidak sengaja upload file yang seharusnya di-ignore:

```bash
# Hapus dari Git tracking (file tetap ada lokal)
git rm --cached nama_file.txt
git rm --cached -r folder_name/

# Commit
git commit -m "Remove unnecessary files"

# Push
git push
```

## Troubleshooting

### Error: "failed to push some refs"
```bash
# Pull dulu untuk sync
git pull origin main --rebase

# Lalu push lagi
git push origin main
```

### Error: "src refspec main does not match any"
```bash
# Coba ganti main dengan master
git push -u origin master
```

### File terlalu besar (>100MB)
- Jangan upload file >100MB
- Untuk model/dataset besar, gunakan Git LFS atau simpan di cloud storage

### Lupa file di .gitignore
```bash
# Tambahkan ke .gitignore
echo "nama_file_atau_folder" >> .gitignore

# Remove dari Git tracking
git rm --cached nama_file
git commit -m "Update .gitignore"
git push
```

## Commands Cheat Sheet

```bash
# Setup
git init
git remote add origin <url>

# Daily workflow
git status                    # Cek status
git add .                     # Stage semua
git add <file>                # Stage file spesifik
git commit -m "message"       # Commit
git push                      # Push ke GitHub
git pull                      # Pull dari GitHub

# Branch
git branch                    # Lihat branch
git branch <name>             # Buat branch baru
git checkout <name>           # Pindah branch
git merge <branch>            # Merge branch

# History
git log                       # Lihat history
git log --oneline             # History singkat

# Undo
git reset HEAD <file>         # Unstage file
git checkout -- <file>        # Undo perubahan
```

## Struktur Commit Message yang Baik

```bash
# Format
git commit -m "Type: Short description"

# Contoh
git commit -m "feat: Add MobileNetV2 training script"
git commit -m "fix: Fix data preprocessing bug"
git commit -m "docs: Update README with results"
git commit -m "refactor: Improve model architecture"
git commit -m "test: Add unit tests for model"
```

**Types:**
- `feat`: Fitur baru
- `fix`: Bug fix
- `docs`: Dokumentasi
- `refactor`: Refactoring code
- `test`: Testing
- `chore`: Maintenance

## Selesai!

Project Anda sekarang sudah online di GitHub! 🎉

URL: `https://github.com/YOUR_USERNAME/face-mask-detection`

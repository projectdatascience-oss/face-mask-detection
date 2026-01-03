# Contributing to Face Mask Detection

Terima kasih atas minat Anda untuk berkontribusi pada project ini! 🎉

## Cara Berkontribusi

### Melaporkan Bug
Jika Anda menemukan bug, silakan buat issue dengan informasi berikut:
- Deskripsi bug yang jelas
- Langkah-langkah untuk mereproduksi bug
- Hasil yang diharapkan vs hasil yang didapat
- Screenshots (jika relevan)
- Environment (OS, Python version, library versions)

### Mengajukan Feature Request
Kami terbuka untuk ide-ide baru! Silakan buat issue dengan:
- Deskripsi feature yang jelas
- Alasan mengapa feature ini berguna
- Contoh penggunaan (jika ada)

### Pull Request Process

1. **Fork repository** ini
2. **Clone** fork Anda ke komputer lokal:
   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   cd face-mask-detection
   ```

3. **Buat branch** baru untuk perubahan Anda:
   ```bash
   git checkout -b feature/nama-feature
   # atau
   git checkout -b fix/nama-bug
   ```

4. **Buat perubahan** Anda:
   - Ikuti style guide yang ada
   - Tambahkan comments yang jelas
   - Update dokumentasi jika diperlukan

5. **Test** perubahan Anda:
   ```bash
   # Pastikan code Anda berjalan dengan baik
   python train_all_models.py
   python evaluate.py --model mobilenetv2
   python predict.py --model mobilenetv2 --image test_image.jpg
   ```

6. **Commit** perubahan Anda:
   ```bash
   git add .
   git commit -m "feat: add new feature X"
   # atau
   git commit -m "fix: resolve issue #123"
   ```

7. **Push** ke fork Anda:
   ```bash
   git push origin feature/nama-feature
   ```

8. **Buat Pull Request** di GitHub:
   - Berikan deskripsi yang jelas tentang perubahan
   - Reference issue terkait (jika ada)
   - Tambahkan screenshots untuk perubahan UI (jika relevan)

## Commit Message Guidelines

Gunakan format berikut untuk commit message:

```
<type>: <subject>

<body> (opsional)

<footer> (opsional)
```

**Types:**
- `feat`: Feature baru
- `fix`: Bug fix
- `docs`: Perubahan dokumentasi
- `style`: Formatting, missing semi colons, etc (tidak mengubah code)
- `refactor`: Refactoring code
- `test`: Menambah atau update tests
- `chore`: Update build tasks, package manager configs, etc

**Contoh:**
```
feat: add real-time video detection

Implement real-time face mask detection using webcam.
Supports both MobileNetV2 and VGG16 models.

Closes #45
```

## Code Style Guidelines

### Python
- Ikuti [PEP 8](https://pep8.org/) style guide
- Gunakan 4 spaces untuk indentation
- Maximum line length: 100 characters
- Gunakan docstrings untuk functions dan classes
- Type hints direkomendasikan

**Contoh:**
```python
def predict_image(model_path: str, img_path: str) -> tuple:
    """
    Prediksi single image menggunakan model.
    
    Args:
        model_path: Path ke model .h5
        img_path: Path ke gambar input
        
    Returns:
        Tuple berisi (predicted_label, confidence_score)
    """
    # Implementation here
    pass
```

### File Organization
```
face-mask-detection/
├── src/              # Source code
├── data/             # Dataset (not tracked)
├── models/           # Trained models (not tracked)
├── results/          # Training results
├── docs/             # Documentation
├── tests/            # Unit tests
└── notebooks/        # Jupyter notebooks (opsional)
```

## Testing

Sebelum submit Pull Request, pastikan:
- [ ] Code berjalan tanpa error
- [ ] Semua existing tests masih pass
- [ ] Menambahkan tests untuk feature baru
- [ ] Dokumentasi sudah diupdate

## Questions?

Jika ada pertanyaan, silakan:
1. Cek [README.md](README.md) terlebih dahulu
2. Cari di existing issues
3. Buat issue baru dengan label `question`

## Code of Conduct

Kami berkomitmen untuk menyediakan lingkungan yang ramah, aman, dan welcoming untuk semua orang.

### Perilaku yang Diharapkan:
- ✅ Menggunakan bahasa yang ramah dan inklusif
- ✅ Menghormati sudut pandang dan pengalaman yang berbeda
- ✅ Menerima kritik konstruktif dengan baik
- ✅ Fokus pada apa yang terbaik untuk komunitas

### Perilaku yang Tidak Dapat Diterima:
- ❌ Bahasa atau gambar yang bersifat seksual
- ❌ Trolling, komentar menghina, atau serangan personal
- ❌ Harassment publik atau privat
- ❌ Publishing informasi privat orang lain

## License

Dengan berkontribusi, Anda setuju bahwa kontribusi Anda akan dilisensikan di bawah MIT License yang sama dengan project ini.

---

Terima kasih sudah berkontribusi! 🚀

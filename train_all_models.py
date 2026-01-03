import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pathlib

# ==========================================
# 1. KONFIGURASI & LOAD DATA
# ==========================================
DATA_DIR = 'data'  # Pastikan folder 'data' ada di direktori yang sama
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # Ubah ke 10-20 untuk hasil lebih maksimal
LR = 0.0001

print(f"Mendeteksi data di: {DATA_DIR}")
data_dir = pathlib.Path(DATA_DIR)

# Split Data (80% Train, 20% Val/Test)
# Kita menggunakan ImageDataGenerator untuk augmentasi sederhana
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False # Jangan shuffle untuk evaluasi agar urutan label benar
)

class_names = list(train_generator.class_indices.keys())
print(f"Kelas ditemukan: {class_names}")

# ==========================================
# 2. VISUALISASI AWAL
# ==========================================

# A. Class Distribution
def plot_class_distribution():
    counts = [len(os.listdir(os.path.join(DATA_DIR, x))) for x in class_names]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_names, y=counts, palette="viridis")
    plt.title("Class Distribution - Distribusi Dataset")
    plt.ylabel("Jumlah Gambar")
    plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# B. Sample Images
def plot_sample_images():
    images, labels = next(train_generator)
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
    plt.suptitle("Sample Images - Contoh Gambar")
    plt.savefig('results/sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()

# Buat folder results jika belum ada
os.makedirs('results', exist_ok=True)
os.makedirs('results/confusion_matrices', exist_ok=True)
os.makedirs('results/training_history', exist_ok=True)

plot_class_distribution()
plot_sample_images()

# ==========================================
# 3. DEFINISI MODEL
# ==========================================

def build_custom_cnn():
    """Model sederhana buatan sendiri"""
    model = Sequential([
        Input(shape=IMG_SIZE + (3,)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="Custom_CNN")
    return model

def build_mobilenet():
    """Transfer Learning: MobileNetV2 - Ringan & Cepat"""
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False # Bekukan base model
    head = base_model.output
    head = GlobalAveragePooling2D()(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(1, activation="sigmoid")(head)
    return Model(inputs=base_model.input, outputs=head, name="MobileNetV2")

def build_vgg16():
    """Transfer Learning: VGG16 - Berat & Akurat"""
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False
    head = base_model.output
    head = Flatten()(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(1, activation="sigmoid")(head)
    return Model(inputs=base_model.input, outputs=head, name="VGG16")

models_to_train = [build_custom_cnn(), build_mobilenet(), build_vgg16()]
history_dict = {}
evaluation_results = []
predictions_dict = {}

# ==========================================
# 4. TRAINING LOOP
# ==========================================
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

for model in models_to_train:
    print(f"\n{'='*60}")
    print(f"Training Model: {model.name}")
    print(f"{'='*60}")
    
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=LR), metrics=["accuracy"])
    
    # Callback untuk early stopping dan model checkpoint
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, f"{model.name}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    history_dict[model.name] = history
    
    # Simpan model final setelah training selesai
    model_final_path = os.path.join(SAVE_DIR, f"{model.name}_final.h5")
    model.save(model_final_path)
    print(f"✓ Model final disimpan di: {model_final_path}")
    
    # Simpan history training
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join('results/training_history', f"{model.name}_history.csv"), index=False)
    print(f"✓ History training disimpan")
    
    # Simpan parameter count (ukuran model)
    total_params = model.count_params()
    
    # ==========================================
    # 5. EVALUASI MODEL
    # ==========================================
    print(f"\n{'='*60}")
    print(f"Evaluating {model.name}...")
    print(f"{'='*60}")
    
    loss, accuracy = model.evaluate(validation_generator)
    
    # Prediksi untuk visualisasi detail
    validation_generator.reset()
    preds = model.predict(validation_generator)
    y_pred = (preds > 0.5).astype(int).ravel()
    y_true = validation_generator.classes
    
    predictions_dict[model.name] = {'y_true': y_true, 'y_pred': y_pred, 'y_score': preds.ravel()}
    
    # Hitung Precision, Recall, F1
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Print classification report
    print(f"\nClassification Report for {model.name}:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    evaluation_results.append({
        'Model': model.name,
        'Accuracy': accuracy,
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-Score': report['weighted avg']['f1-score'],
        'Params': total_params
    })

# Konversi hasil ke DataFrame
df_results = pd.DataFrame(evaluation_results)
print("\n" + "="*60)
print("HASIL AKHIR PERBANDINGAN MODEL")
print("="*60)
print(df_results.to_string(index=False))
print("="*60)

# Simpan hasil evaluasi ke file
df_results.to_csv('results/evaluation_results.csv', index=False)
print("\n✓ Hasil evaluasi disimpan di: results/evaluation_results.csv")

# ==========================================
# 6. VISUALISASI HASIL PERBANDINGAN
# ==========================================

print("\n" + "="*60)
print("MEMBUAT VISUALISASI HASIL...")
print("="*60)

# A. Confusion Matrix
plt.figure(figsize=(15, 4))
for i, model_name in enumerate(predictions_dict.keys()):
    plt.subplot(1, 3, i+1)
    cm = confusion_matrix(predictions_dict[model_name]['y_true'], predictions_dict[model_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/confusion_matrices/all_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# B. Accuracy Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Accuracy', data=df_results, palette='magma')
plt.title('Accuracy Comparison - Perbandingan Akurasi')
plt.ylim(0, 1.1)
for index, row in df_results.iterrows():
    plt.text(index, row.Accuracy + 0.02, f"{row.Accuracy:.2%}", color='black', ha="center", fontsize=10, fontweight='bold')
plt.ylabel('Accuracy')
plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# C. Precision-Recall-F1 Comparison
df_melted = df_results.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1-Score"], var_name="Metric", value_name="Score")
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Score", hue="Metric", data=df_melted, palette="Set2")
plt.title("Precision, Recall, F1 Comparison")
plt.ylim(0, 1.1)
plt.ylabel('Score')
plt.savefig('results/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# D. Model Size vs Accuracy
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Params', y='Accuracy', data=df_results, hue='Model', s=200, style='Model', palette='deep')
plt.xscale('log') # Skala log karena perbedaan parameter sangat besar
plt.title("Model Size (Params) vs Accuracy")
plt.xlabel("Total Parameters (Log Scale)")
plt.ylabel('Accuracy')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig('results/params_vs_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# E. Training History
plt.figure(figsize=(12, 5))
for model_name, history in history_dict.items():
    plt.plot(history.history['val_accuracy'], marker='o', label=f'{model_name} Val Acc', linewidth=2)
    plt.plot(history.history['accuracy'], linestyle='--', alpha=0.5, label=f'{model_name} Train Acc')

plt.title('Training History - Kurva Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/training_history/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# F. ROC Curve Comparison
plt.figure(figsize=(8, 6))
for model_name, data in predictions_dict.items():
    fpr, tpr, _ = roc_curve(data['y_true'], data['y_score'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('results/roc_curve_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 7. SIMPAN LAPORAN FINAL
# ==========================================

print("\n" + "="*60)
print("MENYIMPAN LAPORAN FINAL...")
print("="*60)

with open('results/evaluation_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("LAPORAN EVALUASI MODEL FACE MASK DETECTION\n")
    f.write("="*80 + "\n\n")
    
    f.write("KONFIGURASI TRAINING:\n")
    f.write(f"- Image Size: {IMG_SIZE}\n")
    f.write(f"- Batch Size: {BATCH_SIZE}\n")
    f.write(f"- Epochs: {EPOCHS}\n")
    f.write(f"- Learning Rate: {LR}\n")
    f.write(f"- Total Training Samples: {train_generator.samples}\n")
    f.write(f"- Total Validation Samples: {validation_generator.samples}\n\n")
    
    f.write("="*80 + "\n")
    f.write("HASIL PERBANDINGAN MODEL\n")
    f.write("="*80 + "\n\n")
    f.write(df_results.to_string(index=False))
    f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write("DETAIL CLASSIFICATION REPORT\n")
    f.write("="*80 + "\n\n")
    
    for model_name, data in predictions_dict.items():
        f.write(f"\n{model_name}:\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(data['y_true'], data['y_pred'], target_names=class_names))
        f.write("\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("KESIMPULAN\n")
    f.write("="*80 + "\n\n")
    
    best_model = df_results.loc[df_results['Accuracy'].idxmax()]
    f.write(f"MODEL TERBAIK: {best_model['Model']}\n")
    f.write(f"- Accuracy: {best_model['Accuracy']:.4f}\n")
    f.write(f"- Precision: {best_model['Precision']:.4f}\n")
    f.write(f"- Recall: {best_model['Recall']:.4f}\n")
    f.write(f"- F1-Score: {best_model['F1-Score']:.4f}\n")
    f.write(f"- Total Parameters: {best_model['Params']:,}\n")
    f.write("\n")
    f.write(f"{best_model['Model']} menunjukkan performa terbaik dengan akurasi {best_model['Accuracy']:.2%}\n")
    f.write(f"dan jumlah parameter yang efisien ({best_model['Params']:,} parameters).\n")

print("✓ Laporan lengkap disimpan di: results/evaluation_report.txt")

print("\n" + "="*60)
print("TRAINING DAN EVALUASI SELESAI!")
print("="*60)
print("\nFile-file yang dihasilkan:")
print("1. Models: models/")
print("2. Training History: results/training_history/")
print("3. Confusion Matrices: results/confusion_matrices/")
print("4. Visualizations: results/")
print("5. Evaluation Report: results/evaluation_report.txt")
print("="*60)

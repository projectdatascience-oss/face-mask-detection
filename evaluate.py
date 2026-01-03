"""
Script untuk evaluasi model yang sudah dilatih
Usage: python evaluate.py --model mobilenetv2
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Konfigurasi
DATA_DIR = 'data'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

def load_data():
    """Load validation data"""
    print("Loading validation data...")
    
    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    validation_generator = test_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    class_names = list(validation_generator.class_indices.keys())
    print(f"Classes found: {class_names}")
    print(f"Total validation samples: {validation_generator.samples}")
    
    return validation_generator, class_names

def evaluate_model(model_path, validation_generator, class_names, model_name):
    """Evaluasi model pada validation set"""
    print(f"\nLoading model dari: {model_path}")
    model = load_model(model_path)
    
    print("Evaluating model...")
    # Evaluate
    loss, accuracy = model.evaluate(validation_generator, verbose=1)
    
    # Predictions
    validation_generator.reset()
    predictions = model.predict(validation_generator, verbose=1)
    y_pred = (predictions > 0.5).astype(int).ravel()
    y_true = validation_generator.classes
    y_score = predictions.ravel()
    
    # Classification Report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    results = {
        'Model': model_name,
        'Loss': loss,
        'Accuracy': accuracy,
        'Precision': report_dict['weighted avg']['precision'],
        'Recall': report_dict['weighted avg']['recall'],
        'F1-Score': report_dict['weighted avg']['f1-score']
    }
    
    return results, cm, y_true, y_pred, y_score

def plot_confusion_matrix(cm, class_names, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy info
    total = np.sum(cm)
    correct = np.trace(cm)
    accuracy = correct / total * 100
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.2f}%', 
             ha='center', transform=plt.gca().transAxes, fontsize=11)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(RESULTS_DIR, 'confusion_matrices', f'{model_name}_confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix disimpan di: {output_path}")
    
    plt.show()

def plot_roc_curve(y_true, y_score, model_name):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve: {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(RESULTS_DIR, f'{model_name}_roc_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve disimpan di: {output_path}")
    
    plt.show()

def save_evaluation_report(results, model_name, class_names, y_true, y_pred):
    """Simpan laporan evaluasi ke file"""
    output_path = os.path.join(RESULTS_DIR, f'{model_name}_evaluation_report.txt')
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"LAPORAN EVALUASI MODEL: {model_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write("HASIL EVALUASI:\n")
        f.write(f"- Loss: {results['Loss']:.4f}\n")
        f.write(f"- Accuracy: {results['Accuracy']:.4f} ({results['Accuracy']*100:.2f}%)\n")
        f.write(f"- Precision: {results['Precision']:.4f}\n")
        f.write(f"- Recall: {results['Recall']:.4f}\n")
        f.write(f"- F1-Score: {results['F1-Score']:.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write("\n")
    
    print(f"✓ Laporan evaluasi disimpan di: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluasi Model Face Mask Detection')
    parser.add_argument('--model', type=str, required=True,
                        choices=['custom_cnn', 'mobilenetv2', 'vgg16'],
                        help='Pilih model: custom_cnn, mobilenetv2, atau vgg16')
    parser.add_argument('--visualize', action='store_true',
                        help='Tampilkan visualisasi hasil evaluasi')
    
    args = parser.parse_args()
    
    # Create results directory if not exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'confusion_matrices'), exist_ok=True)
    
    # Map model name to file
    model_mapping = {
        'custom_cnn': 'Custom_CNN_best.h5',
        'mobilenetv2': 'MobileNetV2_best.h5',
        'vgg16': 'VGG16_best.h5'
    }
    
    model_file = model_mapping[args.model]
    model_path = os.path.join(MODELS_DIR, model_file)
    
    # Check if model exists
    if not os.path.exists(model_path):
        # Try final model if best model doesn't exist
        model_file = model_file.replace('_best.h5', '_final.h5')
        model_path = os.path.join(MODELS_DIR, model_file)
        
        if not os.path.exists(model_path):
            print(f"Error: Model tidak ditemukan di {model_path}")
            print(f"Pastikan Anda sudah menjalankan training terlebih dahulu.")
            return
    
    # Load data
    validation_generator, class_names = load_data()
    
    # Evaluate model
    print("\n" + "="*80)
    print(f"EVALUATING MODEL: {args.model.upper()}")
    print("="*80)
    
    results, cm, y_true, y_pred, y_score = evaluate_model(
        model_path, validation_generator, class_names, args.model
    )
    
    # Print results
    print("\n" + "="*80)
    print("HASIL EVALUASI")
    print("="*80)
    for key, value in results.items():
        if key == 'Model':
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")
    print("="*80)
    
    # Save report
    save_evaluation_report(results, args.model, class_names, y_true, y_pred)
    
    # Visualize if requested
    if args.visualize:
        print("\nCreating visualizations...")
        plot_confusion_matrix(cm, class_names, args.model)
        plot_roc_curve(y_true, y_score, args.model)
    
    print("\n✓ Evaluasi selesai!")

if __name__ == '__main__':
    main()

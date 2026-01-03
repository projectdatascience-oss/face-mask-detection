"""
Script untuk prediksi gambar menggunakan model yang sudah dilatih
Usage: python predict.py --model mobilenetv2 --image path/to/image.jpg
"""

import argparse
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Konfigurasi
IMG_SIZE = (224, 224)
CLASS_NAMES = ['with_mask', 'without_mask']
MODELS_DIR = 'models'

def load_and_preprocess_image(img_path, target_size=IMG_SIZE):
    """Load dan preprocess gambar untuk prediksi"""
    # Load image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model_path, img_path):
    """Prediksi single image menggunakan model"""
    print(f"\nLoading model dari: {model_path}")
    model = load_model(model_path)
    
    print(f"Loading gambar dari: {img_path}")
    img_array = load_and_preprocess_image(img_path)
    
    print("Melakukan prediksi...")
    prediction = model.predict(img_array, verbose=0)
    
    # Get class and confidence
    confidence = prediction[0][0]
    predicted_class = 1 if confidence > 0.5 else 0
    predicted_label = CLASS_NAMES[predicted_class]
    
    # Adjust confidence for display
    if predicted_class == 0:
        confidence_score = (1 - confidence) * 100
    else:
        confidence_score = confidence * 100
    
    return predicted_label, confidence_score

def visualize_prediction(img_path, predicted_label, confidence):
    """Visualisasi hasil prediksi"""
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    
    # Add prediction text
    color = 'green' if predicted_label == 'with_mask' else 'red'
    title = f'Prediksi: {predicted_label.upper()}\nConfidence: {confidence:.2f}%'
    plt.title(title, fontsize=16, fontweight='bold', color=color)
    plt.axis('off')
    
    # Save result
    output_path = img_path.replace('.', '_predicted.')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Hasil prediksi disimpan di: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Prediksi Face Mask Detection')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['custom_cnn', 'mobilenetv2', 'vgg16'],
                        help='Pilih model: custom_cnn, mobilenetv2, atau vgg16')
    parser.add_argument('--image', type=str, required=True,
                        help='Path ke gambar yang akan diprediksi')
    parser.add_argument('--visualize', action='store_true',
                        help='Tampilkan visualisasi hasil prediksi')
    
    args = parser.parse_args()
    
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
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Gambar tidak ditemukan di {args.image}")
        return
    
    # Predict
    predicted_label, confidence = predict_image(model_path, args.image)
    
    # Print results
    print("\n" + "="*60)
    print("HASIL PREDIKSI")
    print("="*60)
    print(f"Model: {args.model.upper()}")
    print(f"Gambar: {args.image}")
    print(f"Prediksi: {predicted_label.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print("="*60)
    
    # Visualize if requested
    if args.visualize:
        visualize_prediction(args.image, predicted_label, confidence)

if __name__ == '__main__':
    main()

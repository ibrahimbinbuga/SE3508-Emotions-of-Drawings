import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from preprocess import extract_image_features

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, target_size)
    
    img = img / 255.0
    
    return img

def predict_emotion(model_path, image_path, class_labels=None):
    model = tf.keras.models.load_model(model_path)
    
    img = load_and_preprocess_image(image_path)
    
    img_for_features = (img * 255).astype(np.uint8)
    features = extract_image_features(img_for_features)
    
    img_input = np.expand_dims(img, axis=0)
    features_input = np.expand_dims(features, axis=0)
    
    predictions = model.predict([img_input, features_input])
    
    predicted_class_idx = np.argmax(predictions[0])
    
    if class_labels:
        idx_to_class = {v: k for k, v in class_labels.items()}
        predicted_class = idx_to_class[predicted_class_idx]
    else:
        predicted_class = f"Sınıf {predicted_class_idx}"
    
    probabilities = predictions[0] * 100
    
    return predicted_class, probabilities

def visualize_prediction(image_path, predicted_class, probabilities, class_labels=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    features = extract_image_features(img)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title(f'Tahmin: {predicted_class}')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    
    if class_labels:
        idx_to_class = {v: k for k, v in class_labels.items()}
        class_names = [idx_to_class[i] for i in range(len(probabilities))]
    else:
        class_names = [f"Sınıf {i}" for i in range(len(probabilities))]
    
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, probabilities)
    plt.yticks(y_pos, class_names)
    plt.xlabel('Olasılık (%)')
    plt.title('Duygu Tahmini Olasılıkları')
    
    plt.subplot(2, 2, 3)
    feature_names = ['Kırmızı', 'Yeşil', 'Mavi', 'Çizgi Yoğunluğu', 'Yüz Sayısı']
    plt.bar(feature_names, features)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.title('Çıkarılan Görsel Özellikler')
    
    plt.subplot(2, 2, 4)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    plt.imshow(edges, cmap='gray')
    plt.title(f'Kenar Tespiti (Yoğunluk: {features[3]:.4f})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_model_on_directory(model_path, test_dir, class_labels=None):
    class_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    model = tf.keras.models.load_model(model_path)
    
    correct_predictions = 0
    total_images = 0
    
    for class_dir in class_dirs:
        class_path = os.path.join(test_dir, class_dir)
        
        if class_labels:
            true_class_idx = class_labels[class_dir]
        else:
            true_class_idx = class_dirs.index(class_dir)
        
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            image_path = os.path.join(class_path, img_file)
            
            img = load_and_preprocess_image(image_path)
            
            img_for_features = (img * 255).astype(np.uint8)
            features = extract_image_features(img_for_features)
            
            img_input = np.expand_dims(img, axis=0)
            features_input = np.expand_dims(features, axis=0)
            
            predictions = model.predict([img_input, features_input], verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            
            if predicted_class_idx == true_class_idx:
                correct_predictions += 1
            
            total_images += 1
    
    accuracy = correct_predictions / total_images if total_images > 0 else 0
    
    print(f"Test sonuçları:")
    print(f"Toplam görüntü sayısı: {total_images}")
    print(f"Doğru tahmin sayısı: {correct_predictions}")
    print(f"Doğruluk oranı: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

if __name__ == "__main__":
    model_path = "best_emotion_recognition_model.h5"
    test_image = "../../dataset/test/angry/test_image.jpg"
    
    class_labels = {
        'angry': 0,
        'happy': 1,
        'neutral': 2,
        'sad': 3
    }
    
    predicted_class, probabilities = predict_emotion(model_path, test_image, class_labels)
    print(f"Tahmin edilen duygu: {predicted_class}")
    print(f"Olasılıklar: {probabilities}")
    
    visualize_prediction(test_image, predicted_class, probabilities, class_labels)
    
    test_dir = "../../dataset/test"
    accuracy = test_model_on_directory(model_path, test_dir, class_labels)
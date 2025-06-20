import cv2
import numpy as np
import dlib
import os
import matplotlib.pyplot as plt

face_detector = dlib.get_frontal_face_detector()

def extract_image_features(image):
    try:
        if image.ndim == 2 or image.shape[2] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = cv2.resize(image, (224, 224))

        avg_color = np.mean(image, axis=(0, 1)) / 255.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if gray.dtype != np.uint8:
            gray = np.uint8(gray * 255 if np.max(gray) <= 1.0 else gray)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        faces = face_detector(gray)
        face_count = len(faces)

        return np.array([
            avg_color[0],  
            avg_color[1],  
            avg_color[2],  
            edge_density,  
            face_count     
        ])
    except Exception as e:
        print(f"[HATA - extract_image_features] {e}")
        return np.array([0.5, 0.5, 0.5, 0.5, 0])


def visualize_features(image_path, save_path=None):
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        features = extract_image_features(image)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        faces = face_detector(gray)
        image_with_faces = image.copy()
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(image_with_faces, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title("Orijinal Görüntü")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(edges, cmap='gray')
        plt.title(f"Kenarlar (Yoğunluk: {features[3]:.4f})")
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(image_with_faces)
        plt.title(f"Yüz Sayısı: {int(features[4])}")
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.bar(["Kırmızı", "Yeşil", "Mavi", "Çizgi Yoğunluğu", "Yüz Sayısı"], features)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.title("Özellik Vektörü")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"[KAYDEDİLDİ] Görselleştirme: {save_path}")
        else:
            plt.show()

    except Exception as e:
        print(f"[HATA - visualize_features] {e}")


def test_preprocessing(image_dir, save_dir=None):
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in image_files:
        image_path = os.path.join(image_dir, img_name)
        save_path = os.path.join(save_dir, f"features_{os.path.splitext(img_name)[0]}.png") if save_dir else None

        print(f"[İŞLENİYOR] {img_name}")
        visualize_features(image_path, save_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        features = extract_image_features(image)
        print(f"[ÖZELLİKLER] {img_name}: {features}")

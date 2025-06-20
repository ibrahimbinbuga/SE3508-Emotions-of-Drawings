import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

target_total = 500
input_dir = '../../dataset/train/fear'
image_size = (224, 224)
start_index = 284  

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
current_count = len(image_files)
augment_count = target_total - current_count

print(f"Mevcut görsel sayısı: {current_count}")
print(f"Üretilecek yeni görsel sayısı: {augment_count}")

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

generated = 0
save_index = start_index

while generated < augment_count:
    for image_name in image_files:
        img_path = os.path.join(input_dir, image_name)
        img = load_img(img_path, target_size=image_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        aug_iter = datagen.flow(x, batch_size=1)
        aug_img = next(aug_iter)[0].astype(np.uint8)
        new_image = array_to_img(aug_img)
        new_filename = f"fear_image{save_index}.jpeg"
        new_image.save(os.path.join(input_dir, new_filename))

        generated += 1
        save_index += 1

        if generated >= augment_count:
            break

print(f"{generated} yeni görsel başarıyla kaydedildi.")

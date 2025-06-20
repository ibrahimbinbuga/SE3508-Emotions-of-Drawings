import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import extract_image_features
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import json
import os

validation_dir = '../../dataset/train'
validation_split = 0.2
batch_size = 16
target_size = (224, 224)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class CombinedDataGenerator(Sequence):
    def __init__(self, image_generator, **kwargs):
        super().__init__(**kwargs)
        self.image_generator = image_generator
        self.batch_size = image_generator.batch_size

    def __len__(self):
        return len(self.image_generator)

    def __getitem__(self, idx):
        batch_x, batch_y = next(self.image_generator)
        batch_features = np.array([
            extract_image_features((img * 255).astype(np.uint8))
            for img in batch_x
        ])
        batch_features[:, :3] /= 255.0
        batch_features[:, 3] = batch_features[:, 3]
        batch_features[:, 4] = batch_features[:, 4] / 10.0

        return (
            {
                'image_input': batch_x.astype(np.float32),
                'feature_input': batch_features.astype(np.float32)
            },
            batch_y
        )

    def on_epoch_end(self):
        self.image_generator.on_epoch_end()

val_combined_gen = CombinedDataGenerator(val_generator)

model_path = 'final_emotion_recognition_model_vgg16.keras'
model = load_model(model_path)
print("Model başarıyla yüklendi.")

val_predictions = model.predict(val_combined_gen, verbose=1)
val_pred_classes = np.argmax(val_predictions, axis=1)
val_true_classes = val_combined_gen.image_generator.classes[:len(val_pred_classes)]

with open('./model/class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

print("\nClassification Report:")
print(classification_report(val_true_classes, val_pred_classes, target_names=class_names, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(val_true_classes, val_pred_classes))

print("\nÖzet Validation Metrikleri:")
print(f"Accuracy:  {accuracy_score(val_true_classes, val_pred_classes):.4f}")
print(f"Precision: {precision_score(val_true_classes, val_pred_classes, average='weighted'):.4f}")
print(f"Recall:    {recall_score(val_true_classes, val_pred_classes, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(val_true_classes, val_pred_classes, average='weighted'):.4f}")

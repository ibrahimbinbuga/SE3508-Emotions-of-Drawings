import tensorflow as tf
import numpy as np
import json
import os
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import extract_image_features
from tensorflow.keras.utils import Sequence

model_path = 'final_emotion_recognition_model.keras'
data_dir = '../../dataset/train'
batch_size = 16
target_size = (224, 224)
validation_split = 0.2

with open('./model/class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())
num_classes = len(class_indices)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)


class CombinedDataGenerator(Sequence):
    def __init__(self, image_generator):
        self.image_generator = image_generator
        self.batch_size = image_generator.batch_size

    def __len__(self):
        return len(self.image_generator)

    def __getitem__(self, idx):
        batch_x, batch_y = self.image_generator[idx]

        batch_features = np.array([
            extract_image_features((img * 255).astype(np.uint8)) for img in batch_x
        ])
        batch_features[:, :3] /= 255.0

        return (
            {
                'image_input': batch_x.astype(np.float32),
                'feature_input': batch_features.astype(np.float32)
            },
            batch_y
        )

model = tf.keras.models.load_model(model_path)
print("Model yüklendi.")

val_combined_gen = CombinedDataGenerator(val_generator)

y_pred_probs = model.predict(val_combined_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = val_generator.classes[:len(y_pred)]

report = classification_report(
    y_true, y_pred,
    target_names=class_names,
    digits=4,
    output_dict=True
)

print("\n--- Classification Report ---")
for label, metrics in report.items():
    if label in class_names:
        print(f"{label}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1-score']:.4f}")
        print(f"  Support:   {metrics['support']}")

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n--- Overall Validation Metrics ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

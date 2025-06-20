import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json
from preprocess import extract_image_features
from tensorflow.keras.utils import Sequence

train_dir = '../../dataset/train'
validation_split = 0.2

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=validation_split
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

def num_classes_and_save_indices(gen):
    num = len(gen.class_indices)
    print(f"Sınıf sayısı: {num}")
    print(f"Sınıflar: {gen.class_indices}")
    os.makedirs('./model', exist_ok=True)
    with open('./model/class_indices.json', 'w') as f:
        json.dump(gen.class_indices, f)
    return num

num_classes = num_classes_and_save_indices(train_generator)

def create_model(num_classes):
    image_input = Input(shape=(224, 224, 3), name='image_input')
    feature_input = Input(shape=(5,), name='feature_input')
    
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(image_input)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    combined = Concatenate()([x, feature_input])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    
    output = Dense(num_classes, activation='softmax')(combined)
    
    model = Model(inputs={'image_input': image_input, 'feature_input': feature_input}, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class CombinedDataGenerator(Sequence):
    def __init__(self, image_generator, **kwargs):
        super().__init__(**kwargs)
        self.image_generator = image_generator
        self.batch_size = image_generator.batch_size
        
    def __len__(self):
        return len(self.image_generator)
    
    def __getitem__(self, idx):
        batch_x, batch_y = next(self.image_generator)
        
        batch_features = np.array([extract_image_features((img * 255).astype(np.uint8)) for img in batch_x])
        batch_features[:, :3] /= 255.0  

        return (
            {
                'image_input': batch_x.astype(np.float32),
                'feature_input': batch_features.astype(np.float32)
            },
            batch_y
        )
    
    def on_epoch_end(self):
        self.image_generator.on_epoch_end()


model = create_model(num_classes)
model.summary()

train_combined_gen = CombinedDataGenerator(train_generator)
val_combined_gen = CombinedDataGenerator(validation_generator)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_emotion_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

history = model.fit(
    train_combined_gen,
    steps_per_epoch=len(train_combined_gen),
    validation_data=val_combined_gen,
    validation_steps=len(val_combined_gen),
    epochs=50,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

model.save('final_emotion_recognition_model.keras')

print("Model eğitimi başarıyla tamamlandı!")

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np

print("\nModel değerlendirme metrikleri hesaplanıyor...")

val_combined_gen.on_epoch_end()

val_predictions = model.predict(val_combined_gen)
val_pred_classes = np.argmax(val_predictions, axis=1)

val_true_classes = val_combined_gen.image_generator.classes[:len(val_pred_classes)]

class_names = list(val_combined_gen.image_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(val_true_classes, val_pred_classes, target_names=class_names, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(val_true_classes, val_pred_classes))

precision = precision_score(val_true_classes, val_pred_classes, average='weighted')
recall = recall_score(val_true_classes, val_pred_classes, average='weighted')
f1 = f1_score(val_true_classes, val_pred_classes, average='weighted')
accuracy = accuracy_score(val_true_classes, val_pred_classes)

print("\nDetailed Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

best_train_acc = max(history.history['accuracy'])
best_val_acc = max(history.history['val_accuracy'])
best_train_loss = min(history.history['loss'])
best_val_loss = min(history.history['val_loss'])

print("\nTraining Summary:")
print(f"En iyi eğitim doğruluğu: {best_train_acc:.4f}")
print(f"En iyi doğrulama doğruluğu: {best_val_acc:.4f}")
print(f"En düşük eğitim kaybı: {best_train_loss:.4f}")
print(f"En düşük doğrulama kaybı: {best_val_loss:.4f}")
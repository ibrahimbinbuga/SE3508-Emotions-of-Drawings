import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.applications import VGG16
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
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
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

def create_model(num_classes, fine_tune_at=10):
    image_input = Input(shape=(224, 224, 3), name='image_input')
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=image_input)

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    feature_input = Input(shape=(5,), name='feature_input')
    y = Dense(64, activation='relu')(feature_input)
    y = Dropout(0.3)(y)

    combined = Concatenate()([x, y])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs={'image_input': image_input, 'feature_input': feature_input}, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
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

model = create_model(num_classes, fine_tune_at=15)
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
    min_lr=1e-6
)

history = model.fit(
    train_combined_gen,
    steps_per_epoch=len(train_combined_gen),
    validation_data=val_combined_gen,
    validation_steps=len(val_combined_gen),
    epochs=40,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

model.save('final_emotion_recognition_model_vgg16.keras')
print("Model eğitimi başarıyla tamamlandı!")


from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Concatenate

def create_vgg16_feature_model(num_classes):
    image_input = Input(shape=(224, 224, 3), name='image_input')
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=image_input)
    
    for layer in base_model.layers:
        layer.trainable = False

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

    model = Model(inputs=[image_input, feature_input], outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

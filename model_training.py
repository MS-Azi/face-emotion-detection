# model_training.py
# Step 3: Train a CNN to recognize facial emotions

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# =============================
# 1. Define dataset directories
# =============================
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join("fer2013", "train")
test_dir = os.path.join("fer2013", "test")

# ===================================
# 2. Prepare image data generators
# ===================================
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

# ====================================
# 3. Build the CNN model
# ====================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

# ====================================
# 4. Compile the model
# ====================================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ====================================
# 5. Train the model
# ====================================
epochs = 25

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs
)

# ====================================
# 6. Save the trained model
# ====================================
model.save("face_emotionModel.h5")
print("✅ Model trained and saved as face_emotionModel.h5")

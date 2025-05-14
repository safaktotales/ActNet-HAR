
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from actnet_model import build_actnet
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Configurations
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 40  # Update this if using another dataset
EPOCHS = 30
LEARNING_RATE = 0.001

# ✅ Placeholder paths (Update these accordingly)
train_dir = 'path_to_your_dataset/train'
val_dir = 'path_to_your_dataset/val'

# ✅ Data Augmentation and Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ✅ Model Initialization
model = build_actnet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Model Training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# ✅ Evaluation
val_generator.reset()
preds = model.predict(val_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes

# ✅ Confusion Matrix & Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ✅ Save Model
model.save('actnet_model.h5')

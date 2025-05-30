import os
import shutil
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, ResNet50, ResNet101, ResNet152, VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback


# ==== Step 1: Set Paths for Dataset Handling ====
processed_dir = "/content/drive/MyDrive/model_buildin_dataset/processed_dataset"  # Directory with all labeled images
split_base_dir = "/content/split_dataset"  # Output directory for train/test/val split
os.makedirs(split_base_dir, exist_ok=True)


# ==== Step 2: Clear Existing Splits if Present ====
for split in ['train', 'test', 'val']:
    split_path = os.path.join(split_base_dir, split)
    if os.path.exists(split_path):
        shutil.rmtree(split_path)  # Delete old directory
    os.makedirs(split_path)  # Create new empty directory


# ==== Step 3: Split Images per Class into Train/Test/Val ====
train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1

for cls in os.listdir(processed_dir):
    class_path = os.path.join(processed_dir, cls)
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
    train_imgs, testval_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
    test_imgs, val_imgs = train_test_split(testval_imgs, test_size=(val_ratio / (test_ratio + val_ratio)), random_state=42)

    for split_name, split_imgs in zip(['train', 'test', 'val'], [train_imgs, test_imgs, val_imgs]):
        split_class_dir = os.path.join(split_base_dir, split_name, cls)
        os.makedirs(split_class_dir, exist_ok=True)
        for img in split_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(split_class_dir, img))


# ==== Step 4: Define Data Generators for Model Training ====
train_dir = os.path.join(split_base_dir, 'train')
val_dir = os.path.join(split_base_dir, 'val')
test_dir = os.path.join(split_base_dir, 'test')

img_size = (224, 224)
batch_size = 32


# Augment only training images to generalize better
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation on validation/test

train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_generator = val_test_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
test_generator = val_test_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

num_classes = len(train_generator.class_indices)  # Output layer classes


# ==== Step 5: Build Custom Model Function Using Transfer Learning ====
def build_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Reduce feature maps
    x = Dropout(0.5)(x)  # Regularization
    predictions = Dense(num_classes, activation='softmax')(x)  # Final classification
    return Model(inputs=base_model.input, outputs=predictions)


# ==== Step 6: Custom Callback to Stop Training Once 95% Accuracy is Achieved ====
class StopAt95Acc(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")
        if acc is not None and val_acc is not None:
            if acc >= 0.95 or val_acc >= 0.95:
                print(f"\n⛔️ Stopping early as accuracy reached 95%. (Train: {acc:.4f}, Val: {val_acc:.4f})")
                self.model.stop_training = True


# ==== Step 7: Define List of Pretrained Models to Evaluate ====
models_to_train = {
    "DenseNet121": DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "ResNet101": ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "ResNet152": ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "VGG16": VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
}

results = {}


# ==== Step 8: Train and Evaluate All Models ====
for name, base_model in models_to_train.items():
    print(f"\n🧠 Training model: {name}")


    # Phase 1: Train only top layers
    base_model.trainable = False
    model = build_model(base_model)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
        ModelCheckpoint(f"{name}_best_model.h5", monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
        StopAt95Acc()
    ]

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )


    # Phase 2: Fine-tune the full model
    base_model.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )


    # Evaluate and store the final results
    train_loss, train_acc = model.evaluate(train_generator)
    val_loss, val_acc = model.evaluate(val_generator)
    test_loss, test_acc = model.evaluate(test_generator)

    results[name] = {
        'Train Accuracy': round(train_acc * 100, 2),
        'Val Accuracy': round(val_acc * 100, 2),
        'Test Accuracy': round(test_acc * 100, 2)
    }

    print(f"✅ {name} -> Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")


# ==== Step 9: Show Accuracy Comparison ====
results_df = pd.DataFrame(results).T
print("\n📈 Final Model Accuracy Summary:")
print(results_df)
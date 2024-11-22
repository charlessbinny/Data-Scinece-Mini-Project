import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Function to preprocess images: resizing, converting to RGB, and normalizing
def preprocess_images(image_paths, label, target_size):
    processed_images = []
    labels = []
    for img_path in image_paths:
        try:
            # Open image
            img = Image.open(img_path)
            # Convert to RGB
            img = img.convert('RGB')  # Change to RGB
            # Resize the image
            img = img.resize(target_size)
            # Normalize pixel values (0-255 to 0-1)
            img_array = np.array(img) / 255.0
            # Append the processed image and label
            processed_images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    return np.array(processed_images), np.array(labels)

# Paths to the two relevant folders
mi_folder = "C:/Users/benja/Downloads/gwbz3fsgp8-2/ECG Images of CVD"  # MI images
normal_folder = "C:/Users/benja/Downloads/gwbz3fsgp8-2/Diabetes Person ECG"  # Normal images

# Define the target image size for resizing (e.g., 128x128)
target_size = (128, 128)

# Get all image paths from MI and Normal folders
mi_image_paths = [os.path.join(mi_folder, img) for img in os.listdir(mi_folder)]
normal_image_paths = [os.path.join(normal_folder, img) for img in os.listdir(normal_folder)]

# Preprocess the images and assign labels (1 for MI, 0 for Normal)
mi_images, mi_labels = preprocess_images(mi_image_paths, label=1, target_size=target_size)
normal_images, normal_labels = preprocess_images(normal_image_paths, label=0, target_size=target_size)

# Combine MI and Normal images and labels
X = np.concatenate((mi_images, normal_images), axis=0)
y = np.concatenate((mi_labels, normal_labels), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the data generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Fit the data generator on the reshaped training data
datagen.fit(X_train)

# Load a pre-trained VGG16 model and fine-tune it
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Define the new model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (MI vs Normal)
])

# Compile the model with a learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with the learning rate scheduler
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test), callbacks=[reduce_lr])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# Save the retrained model
model.save('my_cnn_model_retrained1_vgg.h5')

# Evaluate the retrained model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Retrained Model Test Accuracy: {accuracy * 100:.2f}%")


import urllib.request
import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

# -------------------------
# 1️⃣ DOWNLOAD DATASET
# -------------------------

url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"
zip_path = "horse-or-human.zip"

if not os.path.exists(zip_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")

# -------------------------
# 2️⃣ EXTRACT DATASET
# -------------------------

extract_dir = "horse-or-human"

if not os.path.exists(extract_dir):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

# -------------------------
# 3️⃣ DEFINE DIRECTORIES
# -------------------------

train_horse_dir = os.path.join(extract_dir, "horses")
train_human_dir = os.path.join(extract_dir, "humans")

print("Total horse images:", len(os.listdir(train_horse_dir)))
print("Total human images:", len(os.listdir(train_human_dir)))

# -------------------------
# 4️⃣ IMAGE GENERATOR
# -------------------------

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    extract_dir,
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

# -------------------------
# 5️⃣ DEFINE MODEL
# -------------------------

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(300, 300, 3)),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# -------------------------
# 6️⃣ COMPILE MODEL
# -------------------------

from tensorflow.keras.optimizers import RMSprop

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

# -------------------------
# 7️⃣ TRAIN MODEL
# -------------------------

print("\nTraining model...\n")

history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1
)

# -------------------------
# 8️⃣ TEST MODEL
# -------------------------

from tensorflow.keras.preprocessing import image

test_dir = "test_images"

print("\nTesting on new images...\n")

for filename in os.listdir(test_dir):

    path = os.path.join(test_dir, filename)

    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)

    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)

    if prediction[0] > 0.5:
        print(f"{filename} → Human ({prediction[0][0]:.4f})")
    else:
        print(f"{filename} → Horse ({prediction[0][0]:.4f})")

# -------------------------
# 🔟 VISUALIZE INTERMEDIATE REPRESENTATIONS (FIXED)
# -------------------------

print("\nVisualizing intermediate representations...\n")

# Force model to build input (important fix)
sample_batch = train_generator[0][0]
_ = model.predict(sample_batch[:1])

# Create visualization model safely
visualization_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in model.layers]
)

# Get image paths
horse_img_files = [os.path.join(train_horse_dir, f) for f in os.listdir(train_horse_dir)]
human_img_files = [os.path.join(train_human_dir, f) for f in os.listdir(train_human_dir)]

# Pick random image
img_path = random.choice(horse_img_files + human_img_files)
print("Selected image:", img_path)

from tensorflow.keras.preprocessing import image

img = image.load_img(img_path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.0

# Get feature maps
feature_maps = visualization_model.predict(x)

# Display feature maps
for layer, feature_map in zip(model.layers, feature_maps):

    if len(feature_map.shape) == 4:  # Only Conv & Pool layers

        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]

        display_grid = np.zeros((size, size * n_features))

        for i in range(n_features):

            channel_image = feature_map[0, :, :, i]

            channel_image -= channel_image.mean()

            if channel_image.std() != 0:
                channel_image /= channel_image.std()

            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[:, i * size:(i + 1) * size] = channel_image

        scale = 20. / n_features

        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer.name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

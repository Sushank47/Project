import tensorflow as tf
import numpy as np

print("TensorFlow Version:", tf.__version__)

# Load dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize
training_images = training_images / 255.0
test_images = test_images / 255.0

# -----------------------
# BASELINE DNN MODEL
# -----------------------

dnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

dnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------
# CNN MODEL
# -----------------------

# Reshape data for CNN (add channel dimension)
training_images_cnn = training_images.reshape(60000, 28, 28, 1)
test_images_cnn = test_images.reshape(10000, 28, 28, 1)

cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

dnn_model.fit(training_images, training_labels, epochs=5)

dnn_loss, dnn_accuracy = dnn_model.evaluate(test_images, test_labels)


cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.fit(training_images_cnn, training_labels, epochs=20)

cnn_loss, cnn_accuracy = cnn_model.evaluate(test_images_cnn, test_labels)

print("\n========== RESULTS ==========")
print(f"DNN Accuracy: {dnn_accuracy*100:.2f}%")
print(f"CNN Accuracy: {cnn_accuracy*100:.2f}%")

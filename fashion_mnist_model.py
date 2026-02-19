import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# Load dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize
training_images = training_images / 255.0
test_images = test_images / 255.0

# Build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
print("Training started...")
model.fit(training_images, training_labels, epochs=5)

# Evaluate
print("Evaluating...")
model.evaluate(test_images, test_labels)

# Predict first test image
classifications = model.predict(test_images)

print("Predicted Label:", np.argmax(classifications[0]))
print("Actual Label:", test_labels[0])

# Show image
plt.imshow(test_images[0])
plt.colorbar()
plt.show()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# Load dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Reshape for CNN (add channel dimension)
training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# Normalize
training_images = training_images / 255.0
test_images = test_images / 255.0

print("Training shape:", training_images.shape)

# -------------------------
# DEFINE CNN MODEL
# -------------------------

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28,1)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Second convolution removed

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])



model.summary()


# -------------------------
# COMPILE & TRAIN
# -------------------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining CNN...")
model.fit(training_images, training_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# -------------------------
# VISUALIZE ACTIVATIONS
# -------------------------

from tensorflow.keras import models

layer_outputs = [model.layers[0].output,  # Conv2D
                 model.layers[1].output]  # MaxPooling

activation_model = models.Model(inputs=model.inputs, outputs=layer_outputs)

FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28
CONVOLUTION_NUMBER = 6

f, axarr = plt.subplots(3,2, figsize=(8,8))

# Predict once per image
f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1,28,28,1))
f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1,28,28,1))
f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1,28,28,1))

for x in range(2):

    axarr[0,x].imshow(f1[x][0,:,:,CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0,x].grid(False)

    axarr[1,x].imshow(f2[x][0,:,:,CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1,x].grid(False)

    axarr[2,x].imshow(f3[x][0,:,:,CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2,x].grid(False)

plt.tight_layout()
plt.show()

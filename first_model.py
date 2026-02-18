import tensorflow as tf
import numpy as np
from tensorflow import keras

# Step 1: Define the model
model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

# Step 2: Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Step 3: Provide the data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Step 4: Train the model
model.fit(xs, ys, epochs=500)

# Step 5: Use the model
prediction = model.predict(np.array([10.0]))
print("Prediction for X=10:", prediction)

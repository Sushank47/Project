import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# LOAD IMAGE (grayscale)
# -------------------------

i = cv2.imread("output.png", 0)

if i is None:
    print("Image not found. Make sure output.png exists in this folder.")
    exit()

# Resize small 28x28 image for better visualization
i = cv2.resize(i, (256, 256))

# Convert to signed integer to avoid overflow during convolution
i = i.astype(np.int32)

print("Image shape:", i.shape)

# -------------------------
# COPY IMAGE
# -------------------------

i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

# -------------------------
# CONVOLUTION FILTER
# -------------------------

# Vertical edge detection filter
filter = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]

weight = 1

# Apply convolution
for x in range(1, size_x - 1):
    for y in range(1, size_y - 1):

        output_pixel = 0

        output_pixel += i[x-1, y-1] * filter[0][0]
        output_pixel += i[x,   y-1] * filter[0][1]
        output_pixel += i[x+1, y-1] * filter[0][2]
        output_pixel += i[x-1, y]   * filter[1][0]
        output_pixel += i[x,   y]   * filter[1][1]
        output_pixel += i[x+1, y]   * filter[1][2]
        output_pixel += i[x-1, y+1] * filter[2][0]
        output_pixel += i[x,   y+1] * filter[2][1]
        output_pixel += i[x+1, y+1] * filter[2][2]

        output_pixel *= weight

        # Clamp result between 0–255
        output_pixel = max(0, min(255, output_pixel))

        i_transformed[x, y] = output_pixel

# -------------------------
# MAX POOLING (2x2)
# -------------------------

new_x = size_x // 2
new_y = size_y // 2

newImage = np.zeros((new_x, new_y), dtype=np.int32)

for x in range(0, size_x - 1, 2):
    for y in range(0, size_y - 1, 2):

        pixels = [
            i_transformed[x, y],
            i_transformed[x+1, y],
            i_transformed[x, y+1],
            i_transformed[x+1, y+1]
        ]

        newImage[x // 2, y // 2] = max(pixels)

# -------------------------
# DISPLAY RESULTS
# -------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(i, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(i_transformed, cmap='gray')
plt.title("After Convolution")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(newImage, cmap='gray')
plt.title("After Max Pooling")
plt.axis('off')

plt.tight_layout()
plt.show()

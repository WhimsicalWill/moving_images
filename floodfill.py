import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill
from PIL import Image, ImageOps

image = Image.open("leaves.jpg")
image_gray = ImageOps.grayscale(image)
colored_np = np.asarray(image)
gray_np = np.asarray(image_gray) # shape: (h, w, c)

h, w, c = colored_np.shape
flood_tolerance = 40
x_samples = 4
y_samples = 4
x_diff = w // (x_samples + 1)
y_diff = h // (y_samples + 1)

fig, ax = plt.subplots(ncols=x_samples, nrows=y_samples, figsize=(10, 5))
curr_plot = 0
for x in range(1, x_samples + 1):
    for y in range(1, y_samples + 1):
        mask = flood(image_gray, (y * y_diff, x * x_diff), tolerance=flood_tolerance)
        flood_img = np.copy(colored_np)
        flood_img[mask] = [0, 255, 0]
        ax[y - 1, x - 1].imshow(flood_img)
        ax[y - 1, x - 1].plot(x * x_diff, y * y_diff, 'wo')  # seed point
        ax[y - 1, x - 1].set_title(f'Flooded Image ({y, x})')
        curr_plot += 1

plt.show()

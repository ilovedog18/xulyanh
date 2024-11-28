import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh dưới dạng grayscale
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\cheetah.png', cv2.IMREAD_GRAYSCALE)

# Cân bằng histogram
equalized_image = cv2.equalizeHist(image)

# Hiển thị hình ảnh
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Hình ảnh gốc")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Histogram gốc")
plt.hist(image.ravel(), bins=256, range=[0, 256])
plt.grid()

plt.subplot(1, 3, 3)
plt.title("Cân bằng Histogram")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh dưới dạng grayscale
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\cycle.jpg', cv2.IMREAD_GRAYSCALE)

# Tạo đối tượng CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Áp dụng CLAHE
clahe_image = clahe.apply(image)

# Hiển thị hình ảnh
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Hình ảnh gốc")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("CLAHE")
plt.imshow(clahe_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Histogram CLAHE")
plt.hist(clahe_image.ravel(), bins=256, range=[0, 256])
plt.grid()

plt.tight_layout()
plt.show()

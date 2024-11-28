import cv2
import matplotlib.pyplot as plt
import numpy as np

# Đọc hình ảnh và chuyển sang grayscale
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\bird_mask.jpg', cv2.IMREAD_GRAYSCALE)

# Áp dụng bộ lọc Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient theo X
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient theo Y
sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Kết hợp các gradient

# Hiển thị hình ảnh
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.title("Gradient X")
plt.imshow(np.abs(sobel_x), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gradient Y")
plt.imshow(np.abs(sobel_y), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Sobel Combined")
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')

plt.show()

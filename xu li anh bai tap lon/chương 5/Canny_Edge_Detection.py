import cv2
import matplotlib.pyplot as plt

# Đọc hình ảnh
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\lena.jpg')

# Áp dụng Canny Edge Detection
low_threshold = 50  # Ngưỡng dưới
high_threshold = 150  # Ngưỡng trên
edges = cv2.Canny(image, low_threshold, high_threshold)

# Hiển thị hình ảnh
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Hình ảnh gốc")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Canny Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()

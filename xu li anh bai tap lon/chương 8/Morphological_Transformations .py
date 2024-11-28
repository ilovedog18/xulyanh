import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\horse.jpg', cv2.IMREAD_GRAYSCALE)

# Áp dụng ngưỡng để chuyển ảnh thành nhị phân
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Tạo kernel
kernel = np.ones((5, 5), np.uint8)

# Thực hiện Erosion
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# Thực hiện Dilation
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Hiển thị kết quả
plt.figure(figsize=(15, 5))

# Ảnh gốc
plt.subplot(1, 3, 1)
plt.title("Ảnh gốc")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Ảnh sau khi Erosion
plt.subplot(1, 3, 2)
plt.title("Ảnh sau Erosion")
plt.imshow(eroded_image, cmap='gray')
plt.axis('off')

# Ảnh sau khi Dilation
plt.subplot(1, 3, 3)
plt.title("Ảnh sau Dilation")
plt.imshow(dilated_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

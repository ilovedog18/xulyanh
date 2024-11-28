import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc hình ảnh
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\fish.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Kích thước hình ảnh
rows, cols, ch = image.shape

# Xác định 4 điểm gốc và điểm đích
pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250], [250, 250]])

# Tính toán ma trận biến đổi phối cảnh
M = cv2.getPerspectiveTransform(pts1, pts2)

# Áp dụng biến đổi phối cảnh
perspective_transformed = cv2.warpPerspective(image_rgb, M, (cols, rows))

# Hiển thị hình ảnh
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Hình ảnh gốc")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Perspective Transformation")
plt.imshow(perspective_transformed)
plt.axis('off')

plt.show()

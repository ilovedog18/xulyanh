import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\flowers.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Chuyển ảnh thành dạng 2D (pixel và màu sắc)
pixel_values = image_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# 1. Áp dụng K-Means Clustering
# Tiêu chí dừng (epsilon = 0.2) và tối đa 100 lần lặp
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3  # Số cụm
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 2. Chuyển đổi tâm cụm thành dạng số nguyên
centers = np.uint8(centers)
labels = labels.flatten()

# 3. Phân đoạn ảnh dựa trên các cụm
segmented_image = centers[labels]
segmented_image = segmented_image.reshape(image_rgb.shape)

# Hiển thị ảnh gốc và ảnh phân đoạn
plt.figure(figsize=(12, 6))

# Ảnh gốc
plt.subplot(1, 2, 1)
plt.title("Ảnh gốc")
plt.imshow(image_rgb)
plt.axis('off')

# Ảnh sau phân đoạn
plt.subplot(1, 2, 2)
plt.title("Ảnh phân đoạn (K-Means)")
plt.imshow(segmented_image)
plt.axis('off')

plt.tight_layout()
plt.show()

# Lưu kết quả phân đoạn
cv2.imwrite(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\output\segmented_flower.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

# In tâm cụm
print("Tâm cụm (màu sắc):")
print(centers)

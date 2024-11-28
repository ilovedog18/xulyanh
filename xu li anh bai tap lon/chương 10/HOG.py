import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color
import os

# Đọc ảnh
image_path = r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\hill.jpg'

# Kiểm tra file tồn tại
if not os.path.exists(image_path):
    print(f"File không tồn tại: {image_path}")
    exit()

# Đọc ảnh
image = cv2.imread(image_path)

# Kiểm tra ảnh có được đọc không
if image is None:
    print(f"Không thể đọc ảnh: {image_path}")
    exit()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Tính HOG
features, hog_image = hog(
    gray_image, 
    orientations=9, 
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), 
    block_norm='L2-Hys', 
    visualize=True
)

# Hiển thị ảnh
plt.figure(figsize=(12, 6))

# Ảnh gốc
plt.subplot(1, 2, 1)
plt.title("Ảnh gốc")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Ảnh HOG
plt.subplot(1, 2, 2)
plt.title("HOG Features")
plt.imshow(hog_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Hiển thị kích thước vector đặc trưng
print(f"Kích thước vector đặc trưng HOG: {features.shape}")
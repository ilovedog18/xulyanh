import cv2
import matplotlib.pyplot as plt

# Đọc hình ảnh từ file
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\bisons.jpg')

# Chuyển đổi từ BGR sang RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize hình ảnh (Giảm kích thước)
resized_small = cv2.resize(image_rgb, (100, 100), interpolation=cv2.INTER_LINEAR)

# Resize hình ảnh (Tăng kích thước)
resized_large = cv2.resize(image_rgb, (500, 500), interpolation=cv2.INTER_CUBIC)

# Hiển thị hình ảnh
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.title("Hình ảnh gốc")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Giảm kích thước")
plt.imshow(resized_small)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Tăng kích thước")
plt.imshow(resized_large)
plt.axis('off')

plt.show()

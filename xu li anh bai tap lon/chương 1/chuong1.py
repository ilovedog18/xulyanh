import cv2
import matplotlib.pyplot as plt

# Đọc hình ảnh từ file
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\apple.png')


# Chuyển đổi hình ảnh sang định dạng RGB (OpenCV đọc theo định dạng BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Chuyển đổi hình ảnh sang đen-trắng
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Hiển thị hình ảnh gốc và hình ảnh đen-trắng
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Hình ảnh gốc')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Hình ảnh Đen-Trắng')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.show()

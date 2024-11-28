import cv2
import matplotlib.pyplot as plt

# Đọc hình ảnh từ file
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\eagle.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Chuyển đổi không gian màu
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Hiển thị hình ảnh
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("RGB (Hình ảnh gốc)")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Grayscale")
plt.imshow(image_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("HSV")
plt.imshow(cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB))
plt.axis('off')

plt.show()

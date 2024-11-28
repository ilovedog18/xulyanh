import cv2
import matplotlib.pyplot as plt

# Đọc hình ảnh từ file
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\circles.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Làm mờ hình ảnh với bộ lọc Gaussian
blurred_gaussian = cv2.GaussianBlur(image_rgb, (15, 15), 0)

# Làm mờ hình ảnh với bộ lọc trung vị
blurred_median = cv2.medianBlur(image_rgb, 15)

# Hiển thị hình ảnh
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Hình ảnh gốc")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gaussian Blur")
plt.imshow(blurred_gaussian)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Median Blur")
plt.imshow(blurred_median)
plt.axis('off')

plt.show()

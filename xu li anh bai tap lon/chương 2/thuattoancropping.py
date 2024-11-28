import cv2
import matplotlib.pyplot as plt

# Đọc hình ảnh từ file
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\zebras.jpg')

# Chuyển đổi từ BGR sang RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Cắt hình ảnh (vùng từ pixel [50:200, 100:300])
cropped_image = image_rgb[50:200, 100:300]

# Hiển thị hình ảnh
plt.figure(figsize=(8, 5))

plt.subplot(1, 2, 1)
plt.title("Hình ảnh gốc")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Hình ảnh đã cắt")
plt.imshow(cropped_image)
plt.axis('off')

plt.show()

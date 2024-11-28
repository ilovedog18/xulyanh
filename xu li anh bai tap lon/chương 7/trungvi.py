import cv2
import matplotlib.pyplot as plt

# Đọc ảnh gốc
image = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\cars.jpg')

# Thêm nhiễu Salt-and-Pepper để thử nghiệm
import numpy as np
def add_salt_and_pepper_noise(image, prob):
    noisy_image = np.copy(image)
    total_pixels = image.size // image.shape[2]
    num_salt = int(prob * total_pixels / 2)
    num_pepper = int(prob * total_pixels / 2)

    # Add salt noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 255

    # Add pepper noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image

noisy_image = add_salt_and_pepper_noise(image, 0.02)

# Áp dụng Median Filter
filtered_image = cv2.medianBlur(noisy_image, 5)

# Hiển thị hình ảnh
plt.figure(figsize=(15, 5))

# Ảnh gốc
plt.subplot(1, 3, 1)
plt.title("Hình ảnh gốc")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Ảnh nhiễu
plt.subplot(1, 3, 2)
plt.title("Hình ảnh có nhiễu")
plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Ảnh sau khi lọc trung vị
plt.subplot(1, 3, 3)
plt.title("Median Filter")
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

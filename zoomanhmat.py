import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Đọc ảnh từ đĩa
lena = mpimg.imread("images.jpg")  # đọc ảnh từ đĩa dưới dạng numpy ndarray

# Tạo bản sao của mảng lena để có thể thay đổi
lena_copy = lena.copy()

# In giá trị pixel tại vị trí (0, 40)
print(lena_copy[0, 40])  # [180 76 83] - giá trị RGB

# Slicing (cắt ảnh) - in giá trị pixel từ một vùng nhỏ
print(lena_copy[10:13, 20:23, 0:1])  # cắt ảnh

# Lấy kích thước của ảnh
lx, ly, _ = lena_copy.shape

# Tạo lưới tọa độ cho mặt nạ
X, Y = np.ogrid[0:lx, 0:ly]

# Tạo mặt nạ (mask) cho vùng tròn ở giữa ảnh
mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4

# Áp dụng mặt nạ, biến các pixel không nằm trong vùng tròn thành màu đen
lena_copy[mask, :] = 0  # áp dụng mặt nạ

# Hiển thị ảnh
plt.figure(figsize=(10, 10))
plt.imshow(lena_copy)
plt.axis('off')  # Tắt trục
plt.show()  # Hiển thị ảnh

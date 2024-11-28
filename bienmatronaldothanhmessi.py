import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

# Đọc ảnh và điều chỉnh giá trị RGB trong khoảng [0, 1]
im1 = mpimg.imread("images.jpg") / 255  # Ảnh đầu tiên (Messi)
im2 = mpimg.imread("images2.jpg") / 255  # Ảnh thứ hai (Ronaldo)

# Thay đổi kích thước của im2 để khớp với im1
im2 = Image.fromarray((im2 * 255).astype('uint8')).resize(im1.shape[1::-1])
im2 = np.array(im2) / 255  # Chuyển đổi lại sang numpy array và điều chỉnh giá trị

# Tạo hình ảnh morphing
i = 1
plt.figure(figsize=(18, 15))
for alpha in np.linspace(0, 1, 20):
    plt.subplot(4, 5, i)
    plt.imshow((1 - alpha) * im1 + alpha * im2)  # Trộn hai ảnh
    plt.axis('off')  # Tắt hiển thị trục
    i += 1

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()  # Hiển thị tất cả hình ảnh

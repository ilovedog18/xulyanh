import cv2
import matplotlib.pyplot as plt

# Đọc hai ảnh
image1 = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\me6.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(r'C:\Users\hoang\Desktop\file code\xu ly anh\xu li anh bai tap lon\images\me8.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Tạo bộ phát hiện ORB
orb = cv2.ORB_create(nfeatures=500)

# 2. Phát hiện và tính toán mô tả đặc trưng
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# 3. Khớp các mô tả đặc trưng
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)  # Sắp xếp theo độ tương tự

# 4. Vẽ các cặp đặc trưng khớp nhau
result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:20], None, flags=2)

# Hiển thị kết quả
plt.figure(figsize=(12, 6))
plt.title("ORB Feature Matching")
plt.imshow(result_image, cmap='gray')
plt.axis('off')
plt.show()

# Thống kê
print(f"Số lượng điểm đặc trưng được phát hiện (ảnh 1): {len(keypoints1)}")
print(f"Số lượng điểm đặc trưng được phát hiện (ảnh 2): {len(keypoints2)}")
print(f"Số lượng điểm khớp: {len(matches)}")

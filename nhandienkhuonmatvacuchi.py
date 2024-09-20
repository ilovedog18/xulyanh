import cv2
import mediapipe as mp

# Khởi tạo Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Mở camera
cap = cv2.VideoCapture(0)

# Sử dụng mô hình nhận diện khuôn mặt và tay
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Không thể lấy hình ảnh từ camera.")
            break
        
        # Chuyển đổi hình ảnh màu sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Nhận diện khuôn mặt
        face_results = face_detection.process(image_rgb)
        
        # Nhận diện tay
        hand_results = hands.process(image_rgb)

        # Vẽ kết quả nhận diện khuôn mặt
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(image, bbox, (255, 0, 0), 2)
                cv2.putText(image, 'Face', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Vẽ kết quả nhận diện tay
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Hiển thị hình ảnh
        cv2.imshow('Face and Hand Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Nhấn 'ESC' để thoát
            break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()

import os
import cv2
import mediapipe as mp
import json

# MediaPipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# 이미지 파일 경로
image_path = 'test.png'
output_image_path = 'accuracy_test_image/accuracy_answer_landmark.png'

# 디렉토리 확인 및 생성
output_image_dir = os.path.dirname(output_image_path)
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

def extract_and_draw_landmarks_from_image(image_path, output_image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True) as holistic:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        if results.pose_landmarks:
            # 바운딩 박스 좌표 추출
            x_min = min([lm.x for lm in results.pose_landmarks.landmark])
            x_max = max([lm.x for lm in results.pose_landmarks.landmark])
            y_min = min([lm.y for lm in results.pose_landmarks.landmark])
            y_max = max([lm.y for lm in results.pose_landmarks.landmark])
            
            # 바운딩 박스 그리기
            cv2.rectangle(image, (int(x_min * width), int(y_min * height)), (int(x_max * width), int(y_max * height)), (0, 255, 0), 2)
            
            # 랜드마크 그리기
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            cv2.imwrite(output_image_path, image)
            
            # 정규화된 랜드마크 좌표 반환
            return [{'x': (lm.x - x_min) / (x_max - x_min), 'y': (lm.y - y_min) / (y_max - y_min), 'z': lm.z} for lm in results.pose_landmarks.landmark]
        else:
            return None

image_landmarks = extract_and_draw_landmarks_from_image(image_path, output_image_path)
if image_landmarks:
    print("이미지 랜드마크 추출 및 표시 완료")
else:
    print("이미지에서 랜드마크를 찾을 수 없습니다")

# 비디오 파일 경로
video_path = 'ppt_data/ppt_test.mp4'

# JSON 및 이미지 파일 저장 디렉토리 설정
output_json_path = 'accuracy_test_video/accuracy_answer.json'
output_video_path = 'accuracy_test_video/accuracy_test_landmark.mp4'

if not os.path.exists(os.path.dirname(output_json_path)):
    os.makedirs(os.path.dirname(output_json_path))

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)
frame_count = 0

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def compare_landmarks(landmarks1, landmarks2, threshold=0.05):
    matching_points = 0
    for lm1, lm2 in zip(landmarks1, landmarks2):
        if abs(lm1['x'] - lm2['x']) <= threshold and abs(lm1['y'] - lm2['y']) <= threshold and abs(lm1['z'] - lm2['z']) <= threshold:
            matching_points += 1
    return matching_points

best_frame = None
best_landmarks = None
best_frame_index = -1
max_matching_points = 0

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        if results.pose_landmarks:
            # 바운딩 박스 좌표 추출
            x_min = min([lm.x for lm in results.pose_landmarks.landmark])
            x_max = max([lm.x for lm in results.pose_landmarks.landmark])
            y_min = min([lm.y for lm in results.pose_landmarks.landmark])
            y_max = max([lm.y for lm in results.pose_landmarks.landmark])
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (int(x_min * width), int(y_min * height)), (int(x_max * width), int(y_max * height)), (0, 255, 0), 2)
            
            # 정규화된 랜드마크 좌표 추출
            video_landmarks = [{'x': (lm.x - x_min) / (x_max - x_min), 'y': (lm.y - y_min) / (y_max - y_min), 'z': lm.z} for lm in results.pose_landmarks.landmark]
            
            matching_points = compare_landmarks(image_landmarks, video_landmarks)
            print(f"프레임 {frame_count}에서 일치하는 랜드마크 수: {matching_points}")
            
            if matching_points > max_matching_points:
                max_matching_points = matching_points
                best_frame = frame.copy()
                best_landmarks = video_landmarks.copy()
                best_frame_index = frame_count
        
        # 랜드마크를 프레임에 표시
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
        
        # 프레임 저장
        out.write(frame)
        frame_count += 1

cap.release()
out.release()
print(f"총 처리된 프레임 수: {frame_count}")

# 가장 일치하는 랜드마크가 많은 프레임을 저장
if best_frame is not None:
    cv2.imwrite('answer/answer.png', best_frame)
    print(f"가장 일치하는 프레임을 answer.png로 저장했습니다.")
    # 랜드마크 값을 JSON으로 저장
    with open(output_json_path, 'w') as json_file:
        json.dump({'frame_index': best_frame_index, 'landmarks': best_landmarks}, json_file, indent=2)
    print(f"가장 일치하는 프레임의 랜드마크를 {output_json_path}에 저장했습니다.")
else:
    print("일치하는 프레임을 찾을 수 없습니다.")

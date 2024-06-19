import cv2
import mediapipe as mp
import json
import os

# Mediapipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 랜드마크 데이터 추출 함수
def extract_landmark_data(landmarks, landmark_enum=None):
    if landmark_enum is not None:
        return [{'name': landmark_enum(lm_idx).name, 'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility if hasattr(lm, 'visibility') else None}
                for lm_idx, lm in enumerate(landmarks.landmark)]
    else:
        return [{'index': lm_idx, 'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility if hasattr(lm, 'visibility') else None}
                for lm_idx, lm in enumerate(landmarks.landmark)]

# 디렉토리 설정
output_image_dir = 'video_output_images'
output_json_dir = 'video_output_json'

if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)
if not os.path.exists(output_json_dir):
    os.makedirs(output_json_dir)

# 비디오 파일 열기
video_path = '../mediaPipe/ppt_test.mp4'
cap = cv2.VideoCapture(video_path)
frame_count = 0

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path = 'output_video.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

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
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        landmarks = {}
        if results.pose_landmarks:
            landmarks['pose_landmarks'] = extract_landmark_data(results.pose_landmarks, mp_holistic.PoseLandmark)
        if results.face_landmarks:
            landmarks['face_landmarks'] = extract_landmark_data(results.face_landmarks)
        if results.left_hand_landmarks:
            landmarks['left_hand_landmarks'] = extract_landmark_data(results.left_hand_landmarks, mp_holistic.HandLandmark)
        if results.right_hand_landmarks:
            landmarks['right_hand_landmarks'] = extract_landmark_data(results.right_hand_landmarks, mp_holistic.HandLandmark)

        pose_data = {
            'frame_index': frame_count,
            'landmarks': landmarks
        }

        # JSON 파일 저장
        json_file_path = os.path.join(output_json_dir, f'pose_data_{frame_count}.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(pose_data, json_file, indent=2)

        # 이미지에 랜드마크 그리기
        annotated_image = frame.copy()
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                x = int(landmark.x * annotated_image.shape[1])
                y = int(landmark.y * annotated_image.shape[0])
                cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)

        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # 랜드마크가 표시된 이미지 저장
        image_file_path = os.path.join(output_image_dir, f'annotated_image_{frame_count}.png')
        cv2.imwrite(image_file_path, annotated_image)

        # 비디오 파일에 프레임 추가
        out.write(annotated_image)

        frame_count += 1

        cv2.imshow('MediaPipe Holistic', annotated_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total number of frames processed: {frame_count}")

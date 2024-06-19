import cv2
import mediapipe as mp
import json
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

IMAGE_FILES = ["../singlePerson_image/yoo/balance.jpeg"]
BG_COLOR = (192, 192, 192)

landmarks_data = []

def extract_landmark_data(landmarks, landmark_enum=None):
    if landmark_enum is not None:
        return [{'name': landmark_enum(lm_idx).name, 'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility if hasattr(lm, 'visibility') else None}
                for lm_idx, lm in enumerate(landmarks.landmark)]
    else:
        return [{'index': lm_idx, 'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility if hasattr(lm, 'visibility') else None}
                for lm_idx, lm in enumerate(landmarks.landmark)]

with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True) as holistic:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        landmarks = {}
        if results.pose_landmarks:
            landmarks['pose_landmarks'] = extract_landmark_data(results.pose_landmarks, mp_holistic.PoseLandmark)
        if results.face_landmarks:
            landmarks['face_landmarks'] = extract_landmark_data(results.face_landmarks)
        if results.left_hand_landmarks:
            landmarks['left_hand_landmarks'] = extract_landmark_data(results.left_hand_landmarks, mp_holistic.HandLandmark)
        if results.right_hand_landmarks:
            landmarks['right_hand_landmarks'] = extract_landmark_data(results.right_hand_landmarks, mp_holistic.HandLandmark)

        landmarks_data.append({
            'file': file,
            'landmarks': landmarks
        })

        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)

        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        fig = plt.figure(figsize=[10, 10])
        plt.title("Output")
        plt.axis('off')
        plt.imshow(annotated_image[:, :, ::-1])
        plt.show()

with open('image_landmarks_data.json', 'w') as f:
    json.dump(landmarks_data, f, indent=4)

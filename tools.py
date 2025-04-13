import cv2
import mediapipe as mp
import numpy as np


def get_pose_landmarks(video_path, demonstration=False):
    # Инициализация MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils

    global gl_root_y, gl_root_x, gl_root_z
    cap = cv2.VideoCapture(video_path)
    frames_landmarks = []
    scale_factor = 100
    frame = 0

    landmark_names = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index"
    ]

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Преобразуем изображение в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)[0]

        frame_landmarks = {}
        if results.pose_world_landmarks:
            if demonstration:
                mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if frame == 0:
                max_y = max(results.pose_world_landmarks.landmark[29].y, results.pose_world_landmarks.landmark[30].y)

                for ind, landmark in enumerate(results.pose_world_landmarks.landmark):
                        max_y = max(max_y, landmark.y)

                gl_root_y = max_y
                gl_root_x = (results.pose_world_landmarks.landmark[23].x +
                             results.pose_world_landmarks.landmark[24].x) / 2
                gl_root_z = (results.pose_world_landmarks.landmark[23].z +
                             results.pose_world_landmarks.landmark[24].z) / 2
                frame_landmarks['root'] = np.array([0, 0, 0])
            else:
                root_x = (results.pose_world_landmarks.landmark[23].x +
                          results.pose_world_landmarks.landmark[24].x) / 2
                root_z = (results.pose_world_landmarks.landmark[23].z +
                          results.pose_world_landmarks.landmark[24].z) / 2
                frame_landmarks['root'] = np.array([-(root_x - gl_root_x) * scale_factor,
                                                    0,
                                                    -(root_z - gl_root_z) * scale_factor])

            hip_x = (results.pose_world_landmarks.landmark[23].x +
                     results.pose_world_landmarks.landmark[24].x) / 2
            hip_y = (results.pose_world_landmarks.landmark[23].y +
                     results.pose_world_landmarks.landmark[24].y) / 2
            hip_z = (results.pose_world_landmarks.landmark[23].z +
                     results.pose_world_landmarks.landmark[24].z) / 2

            had_c_x = (results.pose_world_landmarks.landmark[7].x +
                       results.pose_world_landmarks.landmark[8].x) / 2
            had_c_y = (results.pose_world_landmarks.landmark[7].y +
                       results.pose_world_landmarks.landmark[8].y) / 2
            had_c_z = (results.pose_world_landmarks.landmark[7].z +
                       results.pose_world_landmarks.landmark[8].z) / 2

            chest_x = (results.pose_world_landmarks.landmark[11].x +
                       results.pose_world_landmarks.landmark[12].x) / 2
            chest_y = (results.pose_world_landmarks.landmark[11].y +
                       results.pose_world_landmarks.landmark[12].y) / 2
            chest_z = (results.pose_world_landmarks.landmark[11].z +
                       results.pose_world_landmarks.landmark[12].z) / 2

            # mid_spine_x = (results.pose_world_landmarks.landmark[11].x +
            #            results.pose_world_landmarks.landmark[12].x) / 2
            # mid_spine_y = (results.pose_world_landmarks.landmark[11].y +
            #            results.pose_world_landmarks.landmark[12].y) / 2
            # mid_spine_z = (results.pose_world_landmarks.landmark[11].z +
            #            results.pose_world_landmarks.landmark[12].z) / 2

            frame_landmarks['hip'] = np.array([-(hip_x - gl_root_x) * scale_factor,
                                                -(hip_y - gl_root_y) * scale_factor,
                                                -(hip_z - gl_root_z) * scale_factor])
            frame_landmarks['had_center'] = np.array([-(had_c_x - gl_root_x) * scale_factor,
                                                      -(had_c_y - gl_root_y) * scale_factor,
                                                      -(had_c_z - gl_root_z) * scale_factor])
            frame_landmarks['chest'] = np.array([-(chest_x - gl_root_x) * scale_factor,
                                                 -(chest_y - gl_root_y) * scale_factor,
                                                 -(chest_z - gl_root_z) * scale_factor])

            for ind, landmark in enumerate(results.pose_world_landmarks.landmark):
                frame_landmarks[landmark_names[ind]] = np.array([-(landmark.x - gl_root_x) * scale_factor,
                                                                 -(landmark.y - gl_root_y) * scale_factor,
                                                                 -(landmark.z - gl_root_z) * scale_factor])
            frames_landmarks.append(frame_landmarks)
        if demonstration:
            image_resized = cv2.resize(image, (800, 600))  # Изменение размера окна
            cv2.imshow("Pose Tracking", image_resized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame += 1
    cap.release()

    print(f"Обработано {len(frames_landmarks)} кадров")
    return frames_landmarks


if __name__ == '__main__':
    i = input("Введите номер видео: ")
    video_path = f"videos/video{i}.mp4"
    landmarks_data = get_pose_landmarks(video_path, True)

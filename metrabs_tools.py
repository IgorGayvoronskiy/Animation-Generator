# import sys
# from tqdm import tqdm
# import time
# import urllib.request
# import json
#
# import cameralib
# import cv2
import numpy as np
# import tensorflow as tf
import json


# def draw_skeleton(image, joints, edges, color=(0, 255, 0)):
#     for i, j in edges:
#         pt1 = tuple(joints[i, :2].astype(int))
#         pt2 = tuple(joints[j, :2].astype(int))
#         cv2.line(image, pt1, pt2, color, 2)
#     return image
#
#
# def load_frames_opencv(path):
#     cap = cv2.VideoCapture(path)
#     frames = []
#     imshape = None
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if imshape is None:
#             imshape = frame.shape[:2]
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(tf.convert_to_tensor(frame_rgb, dtype=tf.uint8))
#     cap.release()
#     return tf.data.Dataset.from_tensor_slices(frames).batch(8).prefetch(1), imshape
#
#
# def get_data_from_video(video_filepath, demonstration=False):
#     # model = tfhub.load('https://bit.ly/metrabs_s')
#
#     skeleton = 'smpl+head_30'
#     joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
#     joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()
#
#     frame_batches, imshape = load_frames_opencv(video_filepath)
#
#     camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=imshape)
#
#     # Видео вывод
#     if (demonstration):
#         video_name = video_filepath.split('/')[-1]
#         out_path = f'/marked_video/{video_name}'
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out_writer = cv2.VideoWriter(out_path, fourcc, 25, (imshape[1], imshape[0]))
#
#     # Данные 3D-точек
#     all_people_poses = []
#     max_people = 0
#
#     start = time.time()
#     print('Start')
#
#     num_frames = 0
#
#     for frame_batch in tqdm(frame_batches):
#         pred = model.detect_poses_batched(
#             frame_batch,
#             intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
#             skeleton=skeleton
#         )
#
#         for frame, boxes, poses3d in zip(frame_batch, pred['boxes'], pred['poses3d']):
#             frame_np = frame.numpy()
#             poses3d_np = poses3d.numpy()
#             num_people = poses3d_np.shape[0]
#
#             # Обновляем список, если увидели новых людей
#             if num_people > max_people:
#                 for _ in range(num_people - max_people):
#                     all_people_poses.append([None] * num_frames)
#                 max_people = num_people
#
#             # Добавляем позы
#             for i in range(max_people):
#                 if i < num_people:
#                     person_pose = poses3d_np[i].tolist()
#                 else:
#                     person_pose = None  # или [[0, 0, 0]] * num_joints
#                 all_people_poses[i].append(person_pose)
#
#             # Визуализация только первого
#             if demonstration and num_people > 0:
#                 projected = camera.world_to_image(poses3d_np)[0]
#                 frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
#                 frame_skel = draw_skeleton(frame_bgr, projected, joint_edges)
#                 out_writer.write(frame_skel)
#             num_frames += 1
#
#     if demonstration:
#         out_writer.release()
#
#     data = data_processing(all_people_poses)
#     return data

def get_data_from_json(json_filepath):
    with open(json_filepath) as f:
        data = np.array(json.load(f))  # (1, 580, 30, 3)
    data = data_processing(data)
    return data


def data_processing(data):
    gl_root_y, gl_root_x, gl_root_z = 0, -10000, 0
    processed_data = []
    scale_factor = 1/10

    landmark_names = ['hip', 'left_hip', 'right_hip', 'bell', 'left_knee', 'right_knee', 'spin', 'left_ankle',
                      'right_ankle', 'thor', 'left_foot_index', 'right_foot_index', 'neck', 'left_collarbone',
                      'right_collarbone', 'head_bot', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                      'left_wrist', 'right_wrist', 'left_han', 'right_han', 'nose', 'left_eye', 'left_ear', 'right_eye',
                      'right_ear', 'head_top']

    key_pers_ind = 0
    for p_ind, person_frames in enumerate(data):
        if person_frames[0] is not None:
            for point in person_frames[0]:
                if point[1] > gl_root_y:
                    gl_root_y = point[1]
                    key_pers_ind = p_ind
    gl_root_x, gl_root_z = data[key_pers_ind][0][0][0], data[key_pers_ind][0][0][2]

    for person_frames in data:
        processed_frames_for_one_person = []
        for frame_landmarks in person_frames:
            processed_landmark4frame = {}
            if frame_landmarks is not None:
                for ind, point in enumerate(frame_landmarks):
                    processed_landmark4frame[landmark_names[ind]] = np.array([-(point[0] - gl_root_x) * scale_factor,
                                                                              -(point[1] - gl_root_y) * scale_factor,
                                                                              -(point[2] - gl_root_z) * scale_factor])
                processed_landmark4frame['head_center'] = (processed_landmark4frame['head_bot'] +
                                                           processed_landmark4frame['head_top']) / 2
                processed_landmark4frame['root'] = processed_landmark4frame['hip'].copy()
                processed_landmark4frame['root'][1] = 0
                processed_landmark4frame['rotated_hip'] = processed_landmark4frame['hip']
                processed_frames_for_one_person.append(processed_landmark4frame)
            else:
                processed_frames_for_one_person.append(None)
        processed_data.append(processed_frames_for_one_person)
    return processed_data


if __name__ == '__main__':
    get_data_from_json('json_files/video7.json')

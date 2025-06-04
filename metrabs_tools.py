import sys
from tqdm import tqdm
import time

from main import mp, tf, tfhub

# !pip install git+https://github.com/isarandi/cameralib

import cv2
import numpy as np
import json
import cameralib


def load_frames_opencv(path):
    cap = cv2.VideoCapture(path)
    frames = []
    imshape = None
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    res_mdp = []
    mdp_marks = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if imshape is None:
            imshape = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)[0]
        landmark = results.pose_world_landmarks.landmark
        mdp_marks['left_wrist'] = [landmark[15].x * 100, -landmark[15].y * 100, -landmark[15].z * 100]
        mdp_marks['right_wrist'] = [landmark[16].x * 100, -landmark[16].y * 100, -landmark[16].z * 100]
        mdp_marks['left_pinky'] = [landmark[17].x * 100, -landmark[17].y * 100, -landmark[17].z * 100]
        mdp_marks['right_pinky'] = [landmark[18].x * 100, -landmark[18].y * 100, -landmark[18].z * 100]
        mdp_marks['left_index'] = [landmark[19].x * 100, -landmark[19].y * 100, -landmark[19].z * 100]
        mdp_marks['right_index'] = [landmark[20].x * 100, -landmark[20].y * 100, -landmark[20].z * 100]
        mdp_marks['left_thumb'] = [landmark[21].x * 100, -landmark[21].y * 100, -landmark[21].z * 100]
        mdp_marks['right_thumb'] = [landmark[22].x * 100, -landmark[22].y * 100, -landmark[22].z * 100]
        frames.append(tf.convert_to_tensor(frame_rgb, dtype=tf.uint8))
        res_mdp.append(mdp_marks)
    cap.release()
    return tf.data.Dataset.from_tensor_slices(frames).batch(8).prefetch(1), imshape, res_mdp


def get_data_from_video(video_filepath, progress_callback=None):
    print('Model_loading')
    if progress_callback:
        progress_callback(0, 10, 'Загрузка модели')
    model = tfhub.load('metrabs_models/metrabs_s')

    skeleton = 'smpl+head_30'
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    frame_batches, imshape, hands_marks = load_frames_opencv(video_filepath)

    camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=imshape)

    # Данные 3D-точек
    all_people_poses = []
    max_people = 0

    start = time.time()
    print('Start')

    num_frames = 0

    count_batches = 0
    total = len(frame_batches)
    for frame_batch in tqdm(frame_batches):
        pred = model.detect_poses_batched(
            frame_batch,
            intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
            skeleton=skeleton
        )

        for frame, boxes, poses3d in zip(frame_batch, pred['boxes'], pred['poses3d']):
            frame_np = frame.numpy()
            poses3d_np = poses3d.numpy()
            num_people = poses3d_np.shape[0]

            # Обновляем список, если увидели новых людей
            if num_people > max_people:
                for _ in range(num_people - max_people):
                    all_people_poses.append([None] * num_frames)
                max_people = num_people

            # Добавляем позы
            for i in range(max_people):
                if i < num_people:
                    person_pose = poses3d_np[i].tolist()
                    person_pose += list(hands_marks[num_frames].values())
                else:
                    person_pose = None  # или [[0, 0, 0]] * num_joints
                all_people_poses[i].append(person_pose)

            num_frames += 1
        if progress_callback:
            progress_callback(count_batches, total, 'Детекция ключевых точек...')

    data = data_processing(all_people_poses)
    return data


def get_video_data_from_json(json_filepath):
    with open(json_filepath) as f:
        data = json.load(f)  # (1, 580, 30, 3)
        res = []
        for person in data:
            for_person = []
            for i, frame in enumerate(person):
                if frame is not None:
                    for_person.append(frame)
            res.append(np.array(for_person))
        data = res
    data = data_processing(data)
    return data


def data_processing(data):
    gl_root_x, gl_root_y, gl_root_z = 0, -1000, 0
    processed_data = []
    scale_factor = 1 / 10

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
        for frame, frame_landmarks in enumerate(person_frames):
            processed_landmark4frame = {}
            # gl_root_y_new = gl_root_y
            # for point in data[key_pers_ind][frame]:
            #     gl_root_y_new = max(gl_root_y_new, point[1])
            if frame_landmarks is not None:
                for ind, point in enumerate(frame_landmarks[:-8]):
                    processed_landmark4frame[landmark_names[ind]] = np.array([(point[0] - gl_root_x) * scale_factor,
                                                                              -(point[1] - gl_root_y) * scale_factor,
                                                                              -(point[2] - gl_root_z) * scale_factor])
                processed_landmark4frame['head_center'] = (processed_landmark4frame['head_bot'] +
                                                           processed_landmark4frame['head_top']) / 2
                processed_landmark4frame['root'] = processed_landmark4frame['hip'].copy()
                processed_landmark4frame['root'][1] = 0
                processed_landmark4frame['rotated_hip'] = processed_landmark4frame['hip']

                processed_landmark4frame['rotated_left_hip'] = processed_landmark4frame['left_hip']
                processed_landmark4frame['rotated_left_knee'] = processed_landmark4frame['left_knee']
                processed_landmark4frame['rotated_left_ankle'] = processed_landmark4frame['left_ankle']
                processed_landmark4frame['rotated_left_foot_index'] = processed_landmark4frame['left_foot_index']

                processed_landmark4frame['rotated_right_hip'] = processed_landmark4frame['right_hip']
                processed_landmark4frame['rotated_right_knee'] = processed_landmark4frame['right_knee']
                processed_landmark4frame['rotated_right_ankle'] = processed_landmark4frame['right_ankle']
                processed_landmark4frame['rotated_right_foot_index'] = processed_landmark4frame['right_foot_index']

                processed_landmark4frame['rotated_left_collarbone'] = processed_landmark4frame['left_collarbone']
                processed_landmark4frame['rotated_left_shoulder'] = processed_landmark4frame['left_shoulder']
                processed_landmark4frame['rotated_left_elbow'] = processed_landmark4frame['left_elbow']
                processed_landmark4frame['rotated_left_wrist'] = processed_landmark4frame['left_wrist']
                processed_landmark4frame['rotated_left_han'] = processed_landmark4frame['left_han']

                processed_landmark4frame['rotated_right_collarbone'] = processed_landmark4frame['right_collarbone']
                processed_landmark4frame['rotated_right_shoulder'] = processed_landmark4frame['right_shoulder']
                processed_landmark4frame['rotated_right_elbow'] = processed_landmark4frame['right_elbow']
                processed_landmark4frame['rotated_right_wrist'] = processed_landmark4frame['right_wrist']
                processed_landmark4frame['rotated_right_han'] = processed_landmark4frame['right_han']

                diff_l = frame_landmarks[-8] - processed_landmark4frame['left_wrist']
                diff_r = frame_landmarks[-7] - processed_landmark4frame['right_wrist']

                processed_landmark4frame['left_pinky'] = frame_landmarks[-6] - diff_l
                processed_landmark4frame['right_pinky'] = frame_landmarks[-5] - diff_r
                processed_landmark4frame['left_index'] = frame_landmarks[-4] - diff_l
                processed_landmark4frame['right_index'] = frame_landmarks[-3] - diff_r
                processed_landmark4frame['left_thumb'] = frame_landmarks[-2] - diff_l
                processed_landmark4frame['right_thumb'] = frame_landmarks[-1] - diff_r

                processed_frames_for_one_person.append(processed_landmark4frame)
            else:
                processed_frames_for_one_person.append(None)
        processed_data.append(processed_frames_for_one_person)
    return processed_data


def get_data_from_image(image_filepath, progress_callback=None):
    if progress_callback:
        progress_callback(0, 10, 'Загрузка модели')
    skeleton_type = 'smpl+head_30'
    model = tfhub.load('metrabs_models/metrabs_s')
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    img = tf.image.decode_image(tf.io.read_file(image_filepath))
    cv2_img = cv2.imread(image_filepath)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    if img.shape[-1] == 4:
        img = img[:, :, :3]
    if progress_callback:
        progress_callback(0, 10, 'Детекция ключевых точек')
    predict = model.detect_poses(img, skeleton=skeleton_type)
    mdp_predict = pose.process(cv2_img)[0]

    mdp_marks = {}

    landmark = mdp_predict.pose_world_landmarks.landmark
    mdp_marks['left_wrist'] = [landmark[15].x * 100, -landmark[15].y * 100, -landmark[15].z * 100]
    mdp_marks['right_wrist'] = [landmark[16].x * 100, -landmark[16].y * 100, -landmark[16].z * 100]
    mdp_marks['left_pinky'] = [landmark[17].x * 100, -landmark[17].y * 100, -landmark[17].z * 100]
    mdp_marks['right_pinky'] = [landmark[18].x * 100, -landmark[18].y * 100, -landmark[18].z * 100]
    mdp_marks['left_index'] = [landmark[19].x * 100, -landmark[19].y * 100, -landmark[19].z * 100]
    mdp_marks['right_index'] = [landmark[20].x * 100, -landmark[20].y * 100, -landmark[20].z * 100]
    mdp_marks['left_thumb'] = [landmark[21].x * 100, -landmark[21].y * 100, -landmark[21].z * 100]
    mdp_marks['right_thumb'] = [landmark[22].x * 100, -landmark[22].y * 100, -landmark[22].z * 100]

    # Данные 3D-точек
    all_people_poses = []

    start = time.time()
    print('Start')

    for boxes, poses3d in zip(predict['boxes'], predict['poses3d']):
        poses3d_np = poses3d.numpy()
        num_people = poses3d_np.shape[0]

        # Добавляем позы
        for i in range(num_people):
            person_pose = poses3d_np[i].tolist()
            all_people_poses.append(person_pose)
        all_people_poses += list(mdp_marks.values())

    print(f"Процесс завершился за : {time.time() - start:.2f}s")

    if len(all_people_poses):
        return True, data_for_bind_pose_processing(np.array(all_people_poses))
    else:
        return False, []


def get_image_data_from_json(json_filepath):
    with open(json_filepath) as f:
        data = json.load(f)  # (1, 580, 30, 3)
        if len(data) > 0:
            data = np.array(data)
            data = data_for_bind_pose_processing(data)
        else:
            return False, []
    return True, data


def data_for_bind_pose_processing(data, scale_k=10):
    gl_root_x, gl_root_y, gl_root_z = 0, -1000, 0
    scale_factor = 1 / scale_k  # надо доработать, чтобы здесь в знаменателе стоял коэффициент, передаваемый в функцию

    landmark_names = ['hip', 'left_hip', 'right_hip', 'bell', 'left_knee', 'right_knee', 'spin', 'left_ankle',
                      'right_ankle', 'thor', 'left_foot_index', 'right_foot_index', 'neck', 'left_collarbone',
                      'right_collarbone', 'head_bot', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                      'left_wrist', 'right_wrist', 'left_han', 'right_han', 'nose', 'left_eye', 'left_ear', 'right_eye',
                      'right_ear', 'head_top']

    for point in data[:-8]:
        gl_root_y = max(gl_root_y, point[1])
    gl_root_x, gl_root_z = data[0][0], data[0][2]
    processed_data = {}
    for ind, point in enumerate(data[:-8]):
        point = np.array(point)
        if point is not None:
            processed_data[landmark_names[ind]] = np.array([(point[0] - gl_root_x) * scale_factor,
                                                            -(point[1] - gl_root_y) * scale_factor,
                                                            -(point[2] - gl_root_z) * scale_factor])
    processed_data['head_center'] = (processed_data['head_bot'] +
                                     processed_data['head_top']) / 2
    processed_data['root'] = processed_data['hip'].copy()
    processed_data['root'][1] = 0
    processed_data['rotated_hip'] = processed_data['hip']

    processed_data['rotated_left_hip'] = processed_data['left_hip']
    processed_data['rotated_left_knee'] = processed_data['left_knee']
    processed_data['rotated_left_ankle'] = processed_data['left_ankle']
    processed_data['rotated_left_foot_index'] = processed_data['left_foot_index']

    processed_data['rotated_right_hip'] = processed_data['right_hip']
    processed_data['rotated_right_knee'] = processed_data['right_knee']
    processed_data['rotated_right_ankle'] = processed_data['right_ankle']
    processed_data['rotated_right_foot_index'] = processed_data['right_foot_index']

    processed_data['rotated_left_collarbone'] = processed_data['left_collarbone']
    processed_data['rotated_left_shoulder'] = processed_data['left_shoulder']
    processed_data['rotated_left_elbow'] = processed_data['left_elbow']
    processed_data['rotated_left_wrist'] = processed_data['left_wrist']
    processed_data['rotated_left_han'] = processed_data['left_han']

    processed_data['rotated_right_collarbone'] = processed_data['right_collarbone']
    processed_data['rotated_right_shoulder'] = processed_data['right_shoulder']
    processed_data['rotated_right_elbow'] = processed_data['right_elbow']
    processed_data['rotated_right_wrist'] = processed_data['right_wrist']
    processed_data['rotated_right_han'] = processed_data['right_han']

    diff_l = data[-8] - processed_data['left_wrist']
    diff_r = data[-7] - processed_data['right_wrist']

    processed_data['left_pinky'] = data[-6] - diff_l
    processed_data['right_pinky'] = data[-5] - diff_r
    processed_data['left_index'] = data[-4] - diff_l
    processed_data['right_index'] = data[-3] - diff_r
    processed_data['left_thumb'] = data[-2] - diff_l
    processed_data['right_thumb'] = data[-1] - diff_r

    return processed_data


if __name__ == '__main__':
    get_data_from_image('C:/Users/djdjd/Downloads/spider_man_bind_pose.png')

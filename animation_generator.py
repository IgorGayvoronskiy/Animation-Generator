import shutil
import math
import fbx
from metrabs_tools import get_data_from_video, get_video_data_from_json
import model_download
import numpy as np

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from useful_classes import resource_path


def create_bone(manager, name, start, end, parent_node, flag=False):
    bone = fbx.FbxSkeleton.Create(manager, name)
    bone.SetSkeletonType(fbx.FbxSkeleton.EType.eLimbNode)
    bone_node = fbx.FbxNode.Create(manager, name)
    bone_node.SetNodeAttribute(bone)

    translation = [end[i] - start[i] for i in range(3)]

    if flag:
        bone_node.LclTranslation.Set(fbx.FbxDouble3(*translation))
    else:
        len_b = math.sqrt(translation[0] ** 2 + translation[1] ** 2 + translation[2] ** 2)
        bone_node.LclTranslation.Set(fbx.FbxDouble3(0, len_b, 0))

    parent_node.AddChild(bone_node)
    return bone_node


def locate_bone(translation, bone_node, flag=False):
    bone_node.LclRotation.Set(fbx.FbxDouble3(0, 0, 0))

    if flag:
        bone_node.LclTranslation.Set(fbx.FbxDouble3(*translation))
    else:
        len_b = math.sqrt(translation[0] ** 2 + translation[1] ** 2 + translation[2] ** 2)
        bone_node.LclTranslation.Set(fbx.FbxDouble3(0, len_b, 0))


def slerp_quaternion(q1, q2, t):
    """Интерполяция между двумя кватернионами"""
    r = R.from_quat([q1, q2])
    return Slerp([0, 1], r)(t).as_quat()


def rotate_bone(name, parent_bone_node, bone_node, swing, twist, angles):
    global_transform = bone_node.EvaluateGlobalTransform()

    quat_fbx = fbx.FbxQuaternion(swing[0], swing[1], swing[2], swing[3])
    rotation_matrix = fbx.FbxAMatrix()
    rotation_matrix.SetQ(quat_fbx)

    twist_fbx = fbx.FbxQuaternion(twist[0], twist[1], twist[2], twist[3])
    twist_matr = fbx.FbxAMatrix()
    twist_matr.SetQ(twist_fbx)

    new_global = rotation_matrix * global_transform

    if parent_bone_node:
        parent_global = parent_bone_node.EvaluateGlobalTransform()
        parent_global_inverse = parent_global.Inverse()
        new_local = parent_global_inverse * new_global
    else:
        new_local = new_global

    if (name.startswith('rotated_') and name not in ['rotated_hip', 'rotated_left_han', 'rotated_right_han',
                                                     'rotated_left_foot_index', 'rotated_right_foot_index']):
        new_local = twist_matr
    else:
        new_local = new_local * twist_matr

    rotation_ = new_local.GetR()

    rotation_ = fbx.FbxDouble3(rotation_[0], rotation_[1], rotation_[2])

    angles[name] = rotation_

    bone_node.SetRotationOrder(fbx.FbxNode.EPivotSet.eSourcePivot, fbx.EFbxRotationOrder.eEulerYXZ)
    bone_node.LclRotation.Set(rotation_)


def project_on_plane(vec, normal):
    normal = normal / np.linalg.norm(normal)
    return vec - np.dot(vec, normal) * normal


def quaternion_twist(curr_x, target_x, axis):
    # Проекция на плоскость, перпендикулярную оси
    proj_curr = project_on_plane(curr_x, axis)
    proj_target = project_on_plane(target_x, axis)

    # Нормализация проекций
    if np.linalg.norm(proj_curr) > 1e-6:
        proj_curr /= np.linalg.norm(proj_curr)
    else:
        proj_curr = np.array([0, 0, 1])
    if np.linalg.norm(proj_target) > 1e-6:
        proj_target /= np.linalg.norm(proj_target)
    else:
        proj_target = np.array([0, 0, 1])

    # Угол между проекциями
    dot = np.clip(np.dot(proj_curr, proj_target), -1.0, 1.0)
    angle = np.arccos(dot)  # в радианах

    # Определение знака угла
    cross = np.cross(proj_curr, proj_target)
    sign = np.sign(np.dot(cross, axis))
    angle *= sign  # применяем знак

    # Кватернион поворота вокруг оси
    quat = R.from_rotvec(axis * angle)
    return quat  # возвращает scipy Rotation object


def fbx_matrix_to_numpy(fbx_matrix):
    mat = np.zeros((4, 4))
    for row in range(4):
        for col in range(4):
            mat[row, col] = fbx_matrix.Get(row, col)
    return mat


def transform_vector(matrix_fbx, vector_np):
    mat_np = fbx_matrix_to_numpy(matrix_fbx)

    vec4 = np.append(vector_np, 0.0)

    transformed = mat_np @ vec4
    return transformed[:3]


def twist_rotate(name, parent_bone_node, frames_landmarks, direction, frame_=0):
    direction = normalize(direction)

    if name == 'hip':
        up_hint = normalize(frames_landmarks[frame_]['left_hip'] - frames_landmarks[frame_]['hip'])
        needed_v = normalize(up_hint)
    elif name == 'rotated_left_hip':
        up_hint = normalize(frames_landmarks[frame_]['left_knee'] - frames_landmarks[frame_]['left_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks[frame_]['left_foot_index'] - frames_landmarks[frame_]['left_ankle'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_hip':
        up_hint = normalize(frames_landmarks[frame_]['right_knee'] - frames_landmarks[frame_]['right_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks[frame_]['right_foot_index'] - frames_landmarks[frame_]['right_ankle'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_knee':
        up_hint = normalize(frames_landmarks[frame_]['left_foot_index'] - frames_landmarks[frame_]['left_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks[frame_]['left_knee'] - frames_landmarks[frame_]['left_hip'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_knee':
        up_hint = normalize(frames_landmarks[frame_]['right_foot_index'] - frames_landmarks[frame_]['right_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks[frame_]['right_knee'] - frames_landmarks[frame_]['right_hip'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_ankle':
        up_hint = normalize(frames_landmarks[frame_]['left_knee'] - frames_landmarks[frame_]['left_ankle'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_ankle':
        up_hint = normalize(frames_landmarks[frame_]['right_knee'] - frames_landmarks[frame_]['right_ankle'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_collarbone':
        up_hint = normalize(frames_landmarks[frame_]['neck'] - frames_landmarks[frame_]['thor'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_collarbone':
        up_hint = normalize(frames_landmarks[frame_]['neck'] - frames_landmarks[frame_]['thor'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_shoulder':
        up_hint = normalize(frames_landmarks[frame_]['left_wrist'] - frames_landmarks[frame_]['left_elbow'])
        needed_v = np.cross(direction, up_hint)
        if abs(np.dot(direction, up_hint)) >= 0.99:
            needed_v = normalize(frames_landmarks[frame_]['left_thumb'] - frames_landmarks[frame_]['left_wrist'])
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_shoulder':
        up_hint = normalize(frames_landmarks[frame_]['right_wrist'] - frames_landmarks[frame_]['right_elbow'])
        needed_v = np.cross(direction, up_hint)
        if abs(np.dot(direction, up_hint)) >= 0.99:
            needed_v = normalize(frames_landmarks[frame_]['right_wrist'] - frames_landmarks[frame_]['right_thumb'])
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_elbow':
        needed_v = normalize(frames_landmarks[frame_]['left_thumb'] - frames_landmarks[frame_]['left_wrist'])
        needed_v = np.cross(direction, needed_v)
        needed_v = normalize(-needed_v)
    elif name == 'rotated_right_elbow':
        needed_v = normalize(frames_landmarks[frame_]['right_wrist'] - frames_landmarks[frame_]['right_thumb'])
        needed_v = np.cross(direction, needed_v)
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_wrist':
        needed_v = normalize(frames_landmarks[frame_]['left_thumb'] - frames_landmarks[frame_]['left_wrist'])
    elif name == 'rotated_right_wrist':
        needed_v = normalize(frames_landmarks[frame_]['right_wrist'] - frames_landmarks[frame_]['right_thumb'])

    parent_global = parent_bone_node.EvaluateGlobalTransform()
    parent_global_inverse = parent_global.Inverse()

    local_needed_v = normalize(transform_vector(parent_global_inverse, needed_v))
    local_direction = np.array([0, 1, 0])

    quat = quaternion_twist(np.array([1, 0, 0]), local_needed_v, local_direction).as_quat()

    return quat


def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def calculate_rotation(base_direction, direction):
    base_direction = normalize(base_direction)
    direction = normalize(direction)

    rotation_ = R.align_vectors([direction], [base_direction])[0]
    quat = rotation_.as_quat()
    return quat


def add_animation(scene, nodes, landmarks_frames, h_l_n, rotations_by_frame, progress_callback=None, frame_rate='30'):
    anim_stack = fbx.FbxAnimStack.Create(scene, "Animation")
    anim_layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
    anim_stack.AddMember(anim_layer)
    scene.SetCurrentAnimationStack(anim_stack)

    total = len(landmarks_frames)

    frame_rate_types = {'24': fbx.FbxTime.EMode.eFrames24, '30': fbx.FbxTime.EMode.eFrames30,
                        '48': fbx.FbxTime.EMode.eFrames48, '60': fbx.FbxTime.EMode.eFrames60,
                        '120': fbx.FbxTime.EMode.eFrames120}

    for frame, landmarks in tqdm(enumerate(landmarks_frames)):
        time = fbx.FbxTime()
        time.SetFrame(frame, frame_rate_types[frame_rate])

        for current_bone_name, current_bone_node in nodes.items():
            x_curve = current_bone_node.LclTranslation.GetCurve(anim_layer, "X", True)
            y_curve = current_bone_node.LclTranslation.GetCurve(anim_layer, "Y", True)
            z_curve = current_bone_node.LclTranslation.GetCurve(anim_layer, "Z", True)

            x_curve.KeyModifyBegin()
            y_curve.KeyModifyBegin()
            z_curve.KeyModifyBegin()

            if current_bone_name == 'root' or current_bone_name == 'hip':
                key_index_x = x_curve.KeyAdd(time)[0]
                key_index_y = y_curve.KeyAdd(time)[0]
                key_index_z = z_curve.KeyAdd(time)[0]

                x_curve.KeySetValue(key_index_x, landmarks_frames[frame][current_bone_name][0])
                y_curve.KeySetValue(key_index_y, landmarks_frames[frame][current_bone_name][1])
                z_curve.KeySetValue(key_index_z, landmarks_frames[frame][current_bone_name][2])

                x_curve.KeySetInterpolation(key_index_x, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
                y_curve.KeySetInterpolation(key_index_y, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
                z_curve.KeySetInterpolation(key_index_z, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)

            x_curve.KeyModifyEnd()
            y_curve.KeyModifyEnd()
            z_curve.KeyModifyEnd()

            x_rot_curve = current_bone_node.LclRotation.GetCurve(anim_layer, "X", True)
            y_rot_curve = current_bone_node.LclRotation.GetCurve(anim_layer, "Y", True)
            z_rot_curve = current_bone_node.LclRotation.GetCurve(anim_layer, "Z", True)

            x_rot_curve.KeyModifyBegin()
            y_rot_curve.KeyModifyBegin()
            z_rot_curve.KeyModifyBegin()

            if (current_bone_name != 'root' and
                    current_bone_name != 'head_top' and current_bone_name not in h_l_n):
                key_index_x = x_rot_curve.KeyAdd(time)[0]
                key_index_y = y_rot_curve.KeyAdd(time)[0]
                key_index_z = z_rot_curve.KeyAdd(time)[0]

                y_rot_curve.KeySetValue(key_index_y, rotations_by_frame[frame][current_bone_name][1])
                x_rot_curve.KeySetValue(key_index_x, rotations_by_frame[frame][current_bone_name][0])
                z_rot_curve.KeySetValue(key_index_z, rotations_by_frame[frame][current_bone_name][2])

                x_rot_curve.KeySetInterpolation(key_index_x,
                                                fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant)
                y_rot_curve.KeySetInterpolation(key_index_y,
                                                fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant)
                z_rot_curve.KeySetInterpolation(key_index_z,
                                                fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant)

            x_rot_curve.KeyModifyEnd()
            y_rot_curve.KeyModifyEnd()
            z_rot_curve.KeyModifyEnd()

        if progress_callback and frame % 100 == 0:
            progress_callback(frame, total, 'Добавление анимации')
    if progress_callback:
        progress_callback(total, total, 'Анимация создана')


def create_animation(key_points_data_path, model_path, from_json=True, progress_callback=None, frame_rate='30',
                     mmodel_path=None):
    print(key_points_data_path, model_path, from_json, frame_rate, mmodel_path)

    if from_json:
        frames_landmarks = get_video_data_from_json(key_points_data_path)[0]
    else:
        frames_landmarks = get_data_from_video(key_points_data_path, progress_callback, mmodel_path)[0]

    if len(frames_landmarks) == 0:
        return None

    bone_structure = {
        'hip': ['root', 'rotated_hip'],
        'rotated_hip': ['hip', 'bell'],
        'bell': ['rotated_hip', 'spin'],
        'spin': ['bell', 'thor'],
        'thor': ['spin', 'neck'],
        'neck': ['thor', 'head_bot'],
        'head_bot': ['neck', 'head_center'],
        'head_center': ['head_bot', 'head_top'],
        'head_top': ['head_center', None],

        'left_hip': ['hip', 'left_knee'],
        'rotated_left_hip': ['left_hip', None],
        'left_knee': ['rotated_left_hip', 'left_ankle'],
        'rotated_left_knee': ['left_knee', None],
        'left_ankle': ['rotated_left_knee', 'left_foot_index'],
        'rotated_left_ankle': ['left_ankle', None],
        'left_foot_index': ['rotated_left_ankle', None],
        'rotated_left_foot_index': ['left_foot_index', None],

        'right_hip': ['hip', 'right_knee'],
        'rotated_right_hip': ['right_hip', None],
        'right_knee': ['rotated_right_hip', 'right_ankle'],
        'rotated_right_knee': ['right_knee', None],
        'right_ankle': ['rotated_right_knee', 'right_foot_index'],
        'rotated_right_ankle': ['right_ankle', None],
        'right_foot_index': ['rotated_right_ankle', None],
        'rotated_right_foot_index': ['right_foot_index', None],

        'left_collarbone': ['thor', 'left_shoulder'],
        'rotated_left_collarbone': ['left_collarbone', None],
        'left_shoulder': ['rotated_left_collarbone', 'left_elbow'],
        'rotated_left_shoulder': ['left_shoulder', None],
        'left_elbow': ['rotated_left_shoulder', 'left_wrist'],
        'rotated_left_elbow': ['left_elbow', None],
        'left_wrist': ['rotated_left_elbow', 'left_han'],
        'rotated_left_wrist': ['left_wrist', None],
        'left_han': ['rotated_left_wrist', None],
        'rotated_left_han': ['left_han', None],

        'right_collarbone': ['thor', 'right_shoulder'],
        'rotated_right_collarbone': ['right_collarbone', None],
        'right_shoulder': ['rotated_right_collarbone', 'right_elbow'],
        'rotated_right_shoulder': ['right_shoulder', None],
        'right_elbow': ['rotated_right_shoulder', 'right_wrist'],
        'rotated_right_elbow': ['right_elbow', None],
        'right_wrist': ['rotated_right_elbow', 'right_han'],
        'rotated_right_wrist': ['right_wrist', None],
        'right_han': ['rotated_right_wrist', None],
        'rotated_right_han': ['right_han', None],

        'nose': ['head_center', None],
        'left_eye': ['head_center', None],
        'right_eye': ['head_center', None],
        'left_ear': ['head_center', None],
        'right_ear': ['head_center', None]
    }

    frame = 0
    head_leaf_nodes = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
    diff_rotated = ['rotated_hip', 'left_hip', 'right_hip', 'left_collarbone', 'right_collarbone']

    manager = fbx.FbxManager.Create()
    scene = fbx.FbxScene.Create(manager, 'MyScene')

    skeleton_root = fbx.FbxSkeleton.Create(manager, 'root')
    skeleton_root.SetSkeletonType(fbx.FbxSkeleton.EType.eRoot)

    root_node = fbx.FbxNode.Create(manager, 'root')
    root_node.SetNodeAttribute(skeleton_root)
    root_node.LclTranslation.Set(fbx.FbxDouble3(*frames_landmarks[frame]['root']))
    scene.GetRootNode().AddChild(root_node)

    nodes = {'root': root_node}
    locations = []
    cur_locations = {'root': frames_landmarks[0]['root']}
    rotations = []
    cur_rotations = {}
    quats = {}

    for current, (parent, daughter) in bone_structure.items():
        flag = False
        parent_node = nodes[parent]
        current_point = frames_landmarks[frame][current]
        parent_point = frames_landmarks[frame][parent]
        if current == 'hip' or current in diff_rotated or current in head_leaf_nodes:
            cur_locations[current] = current_point - parent_point
            flag = True

        nodes[current] = create_bone(manager, current, parent_point, current_point,
                                     parent_node, flag)
    locations.append(cur_locations)

    for current, (parent, daughter) in bone_structure.items():
        parent_node = nodes[parent]
        current_node = nodes[current]
        if daughter is None:
            if current in head_leaf_nodes:
                rotation = calculate_rotation(np.array([0., 1., 0.]),
                                              frames_landmarks[frame][current] - frames_landmarks[frame][parent])
            else:
                rotation = np.array([0, 0, 0, 1], dtype=float)
        else:
            if current == 'hip':
                rotation = calculate_rotation(np.array([0., 1., 0.]),
                                              frames_landmarks[frame][current] - (frames_landmarks[frame]['left_hip'] +
                                                                                  frames_landmarks[frame][
                                                                                      'right_hip']) / 2)
            elif current in diff_rotated:
                rotation = calculate_rotation(frames_landmarks[frame]['hip'] - (frames_landmarks[frame]['left_hip'] +
                                                                                frames_landmarks[frame][
                                                                                    'right_hip']) / 2,
                                              frames_landmarks[frame][daughter] - frames_landmarks[frame][current])
            else:
                rotation = calculate_rotation(frames_landmarks[frame][current] - frames_landmarks[frame][parent],
                                              frames_landmarks[frame][daughter] - frames_landmarks[frame][current])

        twist = np.array([0, 0, 0, 1], dtype=float)
        if ((current.startswith('rotated_') and current not in ['rotated_hip', 'rotated_left_han', 'rotated_right_han',
                                                                'rotated_left_foot_index', 'rotated_right_foot_index'])
                or current == 'hip'):
            if current == 'hip':
                direction = frames_landmarks[0][current] - (frames_landmarks[0]['left_hip'] +
                                                            frames_landmarks[0]['right_hip']) / 2
            else:
                direction = frames_landmarks[0][bone_structure[parent][1]] - frames_landmarks[0][parent]
            twist = twist_rotate(current, parent_node, frames_landmarks, direction)

        rotate_bone(current, parent_node, current_node, rotation, twist, cur_rotations)

        quats[current] = rotation

    rotations.append(cur_rotations)

    total = len(frames_landmarks)

    for frame in tqdm(range(1, len(frames_landmarks))):
        add_frames = [1]
        for t in add_frames:
            cur_locations = {'root': frames_landmarks[frame]['root']}
            cur_rotations = {}
            for current, (parent, daughter) in bone_structure.items():
                flag = False
                current_node = nodes[current]
                bone_location = frames_landmarks[frame][current] - frames_landmarks[frame][parent]
                if current == 'hip':
                    cur_locations[current] = bone_location
                    flag = True
                locate_bone(bone_location, current_node, flag)
            locations.append(cur_locations)
            for current, (parent, daughter) in bone_structure.items():
                parent_node = nodes[parent]
                current_node = nodes[current]
                if daughter is None:
                    if current in head_leaf_nodes:
                        rotation = calculate_rotation(np.array([0., 1., 0.]),
                                                      frames_landmarks[frame][current] - frames_landmarks[frame][
                                                          parent])
                    else:
                        rotation = np.array([0, 0, 0, 1], dtype=float)
                else:
                    if current == 'hip':
                        rotation = calculate_rotation(np.array([0., 1., 0.]),
                                                      frames_landmarks[frame][current] - (
                                                              frames_landmarks[frame]['left_hip'] +
                                                              frames_landmarks[frame][
                                                                  'right_hip']) / 2)
                    elif current in diff_rotated:
                        rotation = calculate_rotation(
                            frames_landmarks[frame]['hip'] - (frames_landmarks[frame]['left_hip'] +
                                                              frames_landmarks[frame]['right_hip']) / 2,
                            frames_landmarks[frame][daughter] - frames_landmarks[frame][current])
                    else:
                        rotation = calculate_rotation(
                            frames_landmarks[frame][current] - frames_landmarks[frame][parent],
                            frames_landmarks[frame][daughter] - frames_landmarks[frame][current])

                if (((daughter is not None or current in head_leaf_nodes)
                     and not current.startswith('rotated_')) or current == 'rotated_hip'):
                    slerp_rotation = slerp_quaternion(quats[current], rotation, t)
                    quats[current] = rotation
                else:
                    slerp_rotation = rotation

                twist = np.array([0, 0, 0, 1], dtype=float)

                if ((current.startswith('rotated_') and current not in ['rotated_hip', 'rotated_left_han',
                                                                        'rotated_right_han', 'rotated_left_foot_index',
                                                                        'rotated_right_foot_index'])
                        or current == 'hip'):
                    if current == 'hip':
                        direction = frames_landmarks[frame][current] - (frames_landmarks[frame]['left_hip'] +
                                                                        frames_landmarks[frame]['right_hip']) / 2
                    else:
                        direction = frames_landmarks[frame][bone_structure[parent][1]] - frames_landmarks[frame][parent]
                    twist = twist_rotate(current, parent_node, frames_landmarks, direction, frame)

                rotate_bone(current, parent_node, current_node, slerp_rotation, twist, cur_rotations)
            rotations.append(cur_rotations)

            if progress_callback:
                progress_callback(frame, total, 'Обработка данных')

    scene2, nodes2 = model_download.get_skeleton_nodes(model_path)
    add_animation(scene2, nodes2, locations, head_leaf_nodes, rotations, progress_callback, frame_rate)

    # Экспорт анимированного FBX
    model_name = model_path.split('/')[-1].split('.')[0]
    video_name = key_points_data_path.split('/')[-1].split('.')[0]
    exporter = fbx.FbxExporter.Create(manager, "")
    exporter.Initialize(resource_path(f"animated_models/animated_{model_name}_from_{video_name}.fbx"), -1,
                        manager.GetIOSettings())
    exporter.Export(scene2)
    exporter.Destroy()
    print(f"FBX анимация создана: animated_{model_name}_from_{video_name}.fbx")

    save_path = resource_path(f"animated_models/animated_{model_name}_from_{video_name}.fbx")

    if not from_json:
        source_path = key_points_data_path
        destination_path = resource_path(f"animated_models/animated_{model_name}_from_{video_name}.mp4")
        shutil.copy2(source_path, destination_path)

    return save_path


if __name__ == '__main__':
    input_type = input('Введи v если видео или j, если json: ')
    if input_type == 'v':
        i = input('Введите номер тестового видео: ')
        create_animation(f'Source/videos/video{i}.mp4',
                         'Source/rigged_models/spider_man_model_G.fbx', False)
    elif input_type == 'j':
        i = input('Введите номер тестового json: ')
        create_animation(f'Source/json_files/video{i}.json',
                         'Source/rigged_models/spider_man_model_G.fbx', True)

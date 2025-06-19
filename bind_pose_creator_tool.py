import math
import fbx
from metrabs_tools import get_data_from_image, get_image_data_from_json
import numpy as np

from scipy.spatial.transform import Rotation as R
from useful_classes import resource_path


def create_bone(manager, name, start, end, parent_node, flag=False, diff=None):
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

    # if diff is not None and name == 'rotated_left_hip':
    #     len_bone = math.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
    #     cur_len = bone_node.LclScaling.Get()
    #     bone_node.LclScaling.Set(fbx.FbxDouble3(cur_len[0], len_bone, cur_len[2]))
    #     print(len_bone, cur_len[1])
    #     print(parent_node.GetName())

    parent_node.AddChild(bone_node)

    return bone_node


def rotate_bone(name, parent_bone_node, bone_node, swing, twist):
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
    return transformed[:3]  # вернём обратно в 3D


def twist_rotate(name, parent_bone_node, frames_landmarks, direction):
    direction = normalize(direction)

    if name == 'hip':
        up_hint = normalize(frames_landmarks['left_hip'] - frames_landmarks['hip'])
        needed_v = normalize(up_hint)
    elif name == 'rotated_left_hip':
        up_hint = normalize(frames_landmarks['left_knee'] - frames_landmarks['left_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks['left_foot_index'] - frames_landmarks['left_ankle'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_hip':
        up_hint = normalize(frames_landmarks['right_knee'] - frames_landmarks['right_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks['right_foot_index'] - frames_landmarks['right_ankle'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_knee':
        up_hint = normalize(frames_landmarks['left_foot_index'] - frames_landmarks['left_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks['left_knee'] - frames_landmarks['left_hip'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_knee':
        up_hint = normalize(frames_landmarks['right_foot_index'] - frames_landmarks['right_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks['right_knee'] - frames_landmarks['right_hip'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_ankle':
        up_hint = normalize(frames_landmarks['left_knee'] - frames_landmarks['left_ankle'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_ankle':
        up_hint = normalize(frames_landmarks['right_knee'] - frames_landmarks['right_ankle'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_collarbone':
        up_hint = normalize(frames_landmarks['neck'] - frames_landmarks['thor'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_collarbone':
        up_hint = normalize(frames_landmarks['neck'] - frames_landmarks['thor'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_shoulder':
        up_hint = normalize(frames_landmarks['left_wrist'] - frames_landmarks['left_elbow'])
        needed_v = np.cross(direction, up_hint)
        if abs(np.dot(direction, up_hint)) >= 0.99:
            needed_v = normalize(frames_landmarks['left_thumb'] - frames_landmarks['left_wrist'])
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_shoulder':
        up_hint = normalize(frames_landmarks['right_wrist'] - frames_landmarks['right_elbow'])
        needed_v = np.cross(direction, up_hint)
        if abs(np.dot(direction, up_hint)) >= 0.99:
            needed_v = normalize(frames_landmarks['right_wrist'] - frames_landmarks['right_thumb'])
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_elbow':
        needed_v = normalize(frames_landmarks['left_thumb'] - frames_landmarks['left_wrist'])
    elif name == 'rotated_right_elbow':
        needed_v = normalize(frames_landmarks['right_wrist'] - frames_landmarks['right_thumb'])
    elif name == 'rotated_left_wrist':
        needed_v = normalize(frames_landmarks['left_thumb'] - frames_landmarks['left_wrist'])
    elif name == 'rotated_right_wrist':
        needed_v = normalize(frames_landmarks['right_wrist'] - frames_landmarks['right_thumb'])

    parent_global = parent_bone_node.EvaluateGlobalTransform()
    parent_global_inverse = parent_global.Inverse()

    local_needed_v = normalize(transform_vector(parent_global_inverse, needed_v))
    local_direction = np.array([0, 1, 0])

    quat = quaternion_twist(np.array([1, 0, 0]), local_needed_v, local_direction).as_quat()

    return quat


def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def build_rotation_matrix(forward, up):
    forward = normalize(forward)
    right = normalize(np.cross(up, forward))
    up_corrected = normalize(np.cross(forward, right))
    rot_matrix = np.column_stack((right, up_corrected, forward))
    if np.linalg.det(rot_matrix) < 0:
        rot_matrix[:, 0] = -rot_matrix[:, 0]
    return rot_matrix


def calculate_rotation3(name, start, cur, end, standard_direction=False):
    direction = normalize(np.array(end, dtype=float) - np.array(cur, dtype=float))

    # Базовый вектор направления
    if standard_direction:
        base_direction = np.array([0, 1, 0], dtype=float)
    else:
        base_direction = normalize(np.array(cur, dtype=float) - np.array(start, dtype=float))

    # Выравниваем основное направление
    align_rot = R.align_vectors([direction], [base_direction])[0]

    up_hint = np.array([1, 0, 0])
    # if abs(np.dot(direction, up_hint)) > 0.9:
    #     up_hint = np.array([0, 0, 1])
    if name == 'left_collarbone' or name == 'right_collarbone':
        up_hint = np.array([0, 0, 1])

    # Повернем базовый up_hint через align_rot — это то, куда он "попал"
    rotated_up = align_rot.apply(up_hint)

    # Целевой up (как вектор, перпендикулярный кости)
    # Правильный up можно задать явно — например, на основе соседних костей
    # Здесь мы просто проецируем исходный up_hint в плоскость, перпендикулярную direction
    proj_up = up_hint - np.dot(up_hint, direction) * direction
    proj_up = normalize(proj_up)

    # То же самое с rotated_up — проекция в ту же плоскость
    proj_rotated_up = rotated_up - np.dot(rotated_up, direction) * direction
    proj_rotated_up = normalize(proj_rotated_up)

    # Угол между ними (twist вокруг оси кости)
    cos_angle = np.clip(np.dot(proj_up, proj_rotated_up), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    cross = np.dot(np.cross(proj_rotated_up, proj_up), direction)
    if cross < 0:
        angle = -angle

    twist_rot = R.from_rotvec(angle * direction)  # Вращение вокруг оси кости

    # Общий поворот: сначала выравниваем кость, потом корректируем twist
    final_rot = twist_rot * align_rot
    return final_rot.as_quat()


def calculate_rotation2(start, cur, end, standard_direction=False):
    direction = np.array(end, dtype=float) - np.array(cur, dtype=float)
    direction = direction / np.linalg.norm(direction)  # Нормализация

    # Начальная ориентация (кость должна смотреть вдоль Z)
    if standard_direction:
        base_direction = np.array([0, 1, 0], dtype=float)
    else:
        base_direction = np.array(cur, dtype=float) - np.array(start, dtype=float)
        base_direction = base_direction / np.linalg.norm(base_direction)

    up_hint = np.array([1, 0, 0])
    base_hint = np.array([1, 0, 0])
    if abs(np.dot(direction, up_hint)) >= 0.85:
        up_hint = np.array([0, 0, 1])
    if abs(np.dot(base_direction, base_hint)) >= 0.85:
        base_hint = np.array([0, 0, 1])
    base_matrix = build_rotation_matrix(base_direction, base_hint)
    target_matrix = build_rotation_matrix(direction, up_hint)

    # Переход от base → target
    rot_matrix = target_matrix @ np.linalg.inv(base_matrix)
    return R.from_matrix(rot_matrix).as_quat()


def calculate_rotation(start, cur, end, standard_direction=False):
    direction = np.array(end, dtype=float) - np.array(cur, dtype=float)
    direction = direction / np.linalg.norm(direction)  # Нормализация

    # Начальная ориентация (кость должна смотреть вдоль Z)
    if standard_direction:
        base_direction = np.array([0, 1, 0], dtype=float)
    else:
        base_direction = np.array(cur, dtype=float) - np.array(start, dtype=float)
        base_direction = base_direction / np.linalg.norm(base_direction)

    # Кватернион поворота
    rotation_ = R.align_vectors([direction], [base_direction])[0]
    quat = rotation_.as_quat()  # [x, y, z, w]
    return quat


def create_bind_pose(file_path, from_json=True, progress_callback=None):
    if from_json:
        is_done, frames_landmarks = get_image_data_from_json(file_path)
        if not is_done:
            return None
    else:
        is_done, frames_landmarks = get_data_from_image(file_path, progress_callback)
        if not is_done:
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

    head_leaf_nodes = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
    diff_rotated = ['rotated_hip', 'left_hip', 'right_hip', 'left_collarbone', 'right_collarbone']
    no_rotate_bone = ['hip', 'rotated_hip', 'bell', 'spin', 'thor', 'neck', 'head_bot', 'head_center', 'head_top']

    rotate_spine = False

    # Создание FBX-сцены
    manager = fbx.FbxManager.Create()
    scene = fbx.FbxScene.Create(manager, 'MyScene')

    # Создаем кости
    skeleton_root = fbx.FbxSkeleton.Create(manager, 'root')
    skeleton_root.SetSkeletonType(fbx.FbxSkeleton.EType.eRoot)

    # Создаем узел скелета
    root_node = fbx.FbxNode.Create(manager, 'root')
    root_node.SetNodeAttribute(skeleton_root)
    root_node.LclTranslation.Set(fbx.FbxDouble3(*frames_landmarks['root']))
    scene.GetRootNode().AddChild(root_node)

    total = len(bone_structure)
    # Добавляем кости в сцену
    nodes = {'root': root_node}
    for ind, (current, (parent, daughter)) in enumerate(bone_structure.items()):
        flag = False
        parent_node = nodes[parent]
        if current == 'left_knee2':
            current_point = frames_landmarks['left_knee']
        else:
            current_point = frames_landmarks[current]
        parent_point = frames_landmarks[parent]

        if current == 'hip' or current in diff_rotated or current in head_leaf_nodes:
            flag = True

        nodes[current] = create_bone(manager, current, parent_point, current_point,
                                     parent_node, flag)

        if progress_callback:
            progress_callback(ind, total, 'Добавление костей')

    for ind, (current, (parent, daughter)) in enumerate(bone_structure.items()):
        parent_node = nodes[parent]
        current_node = nodes[current]
        rotation = np.array([0, 0, 0, 1], dtype=float)
        if daughter is None:
            if current in head_leaf_nodes:
                if current == 'left_eye' or current == 'right_eye' or current == 'nose':
                    rotation = calculate_rotation(frames_landmarks[parent], [0, 0, 0],
                                                  [0, 0, 1], True)
                elif current == 'left_ear':
                    rotation = calculate_rotation(frames_landmarks[parent], [0, 0, 0],
                                                  [1, 0, 0], True)
                elif current == 'right_ear':
                    rotation = calculate_rotation(frames_landmarks[parent], [0, 0, 0],
                                                  [-1, 0, 0], True)
            else:
                rotation = np.array([0, 0, 0, 1], dtype=float)
        else:
            if current == 'hip' or current == 'rotated_left_hip':
                rotation = np.array([0, 0, 0, 1], dtype=float)
            elif current in diff_rotated:
                rotation = calculate_rotation(frames_landmarks[parent], frames_landmarks[current],
                                              frames_landmarks[daughter], True)
            else:
                rotation = calculate_rotation(frames_landmarks[parent], frames_landmarks[current],
                                              frames_landmarks[daughter])
            if not rotate_spine and current in no_rotate_bone:
                rotation = np.array([0, 0, 0, 1], dtype=float)

        twist = np.array([0, 0, 0, 1], dtype=float)
        if ((current.startswith('rotated_') and current not in ['rotated_hip', 'rotated_left_han',
                                                                'rotated_right_han', 'rotated_left_foot_index',
                                                                'rotated_right_foot_index'])
                and current != 'hip'):
            direction = frames_landmarks[bone_structure[parent][1]] - frames_landmarks[parent]
            twist = twist_rotate(current, parent_node, frames_landmarks, direction)

        rotate_bone(current, parent_node, current_node, rotation, twist)

        if progress_callback:
            progress_callback(ind, total, 'Поворот костей')

    # Сохраняем FBX
    exporter = fbx.FbxExporter.Create(manager, "")
    name_file = file_path.split('/')[-1].split('.')[0]
    save_path = resource_path(f'Source/Skeletons/{name_file}.fbx')
    exporter.Initialize(save_path, -1, manager.GetIOSettings())
    exporter.Export(scene)
    exporter.Destroy()
    print(f'FBX файл создан: {name_file}.fbx')

    return save_path


if __name__ == '__main__':
    create_bind_pose("Source/json_files/spider_man_bind_pose.json", True)

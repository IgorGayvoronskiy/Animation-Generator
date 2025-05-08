import math
import sys
import fbx
import tools
import metrabs_tools
import numpy as np
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

path = input('Введите путь до файла с bind позой: ')

is_done, frames_landmarks = metrabs_tools.get_image_data_from_json(path)
if not is_done:
    sys.exit('Некорректные данные')

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
    'left_knee': ['left_hip', 'left_ankle'],
    'left_ankle': ['left_knee', 'left_foot_index'],
    'left_foot_index': ["left_ankle", None],

    'right_hip': ['hip', 'right_knee'],
    'right_knee': ['right_hip', 'right_ankle'],
    'right_ankle': ['right_knee', 'right_foot_index'],
    'right_foot_index': ['right_ankle', None],

    'left_collarbone': ['thor', 'left_shoulder'],
    'left_shoulder': ['left_collarbone', 'left_elbow'],
    'left_elbow': ['left_shoulder', 'left_wrist'],
    'left_wrist': ['left_elbow', 'left_han'],
    'left_han': ['left_wrist', None],

    'right_collarbone': ['thor', 'right_shoulder'],
    'right_shoulder': ['right_collarbone', 'right_elbow'],
    'right_elbow': ['right_shoulder', 'right_wrist'],
    'right_wrist': ['right_elbow', 'right_han'],
    'right_han': ['right_wrist', None],

    'nose': ['head_center', None],
    'left_eye': ['head_center', None],
    'right_eye': ['head_center', None],
    'left_ear': ['head_center', None],
    'right_ear': ['head_center', None]
}

head_leaf_nodes = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
diff_rotated = ['rotated_hip', 'left_hip', 'right_hip', 'left_collarbone', 'right_collarbone']
no_rotate_bone = ['rotated_hip', 'bell', 'spin', 'thor', 'neck', 'head_bot', 'head_center', 'head_top']

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


# Функция создания кости
def create_bone(name, start, end, parent_node, flag=False):
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


def rotate_bone(parent_bone_node, bone_node, quat):
    global_transform = bone_node.EvaluateGlobalTransform()

    quat_fbx = fbx.FbxQuaternion(quat[0], quat[1], quat[2], quat[3])
    rotation_matrix = fbx.FbxAMatrix()
    rotation_matrix.SetQ(quat_fbx)

    # 3. Умножаем глобальную матрицу на вращение (в глобальном пространстве!)
    new_global = rotation_matrix * global_transform

    # 4. Получаем глобальную матрицу родителя
    if parent_bone_node:
        parent_global = parent_bone_node.EvaluateGlobalTransform()
        parent_global_inverse = parent_global.Inverse()
        # 5. Получаем локальную матрицу, которую нужно задать в узел
        new_local = parent_global_inverse * new_global
    else:
        new_local = new_global  # если это root

    # 6. Применяем локальные transform'ы
    rotation_ = new_local.GetR()

    rotation_ = fbx.FbxDouble3(rotation_[0], rotation_[1], rotation_[2])

    bone_node.LclRotation.Set(rotation_)


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


def calculate_rotation(start, cur, end, standard_direction=False):
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


def calculate_rotation2(start, cur, end, standard_direction=False):
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


# Добавляем кости в сцену
nodes = {'root': root_node}
for current, (parent, daughter) in bone_structure.items():
    flag = False
    parent_node = nodes[parent]
    current_point = frames_landmarks[current]
    parent_point = frames_landmarks[parent]
    if current == 'hip' or current in diff_rotated or current in head_leaf_nodes:
        flag = True

    nodes[current] = create_bone(current, parent_point, current_point,
                                 parent_node, flag)

for current, (parent, daughter) in bone_structure.items():
    parent_node = nodes[parent]
    current_node = nodes[current]
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
            rotation = (0, 0, 0, 0)
    else:
        if current == 'hip':
            rotation = (0, 0, 0, 0)
        elif current in diff_rotated:
            rotation = calculate_rotation(frames_landmarks[parent], frames_landmarks[current],
                                          frames_landmarks[daughter], True)
        else:
            rotation = calculate_rotation(frames_landmarks[parent], frames_landmarks[current],
                                          frames_landmarks[daughter])
        if not rotate_spine and current in no_rotate_bone:
            rotation = (0, 0, 0, 0)
    rotate_bone(parent_node, current_node, rotation)

# Сохраняем FBX
exporter = fbx.FbxExporter.Create(manager, "")
name_file = path.split('/')[1].split('.')[0]
exporter.Initialize(f'{name_file}.fbx', -1, manager.GetIOSettings())
exporter.Export(scene)
exporter.Destroy()
print(f'FBX файл создан: {name_file}.fbx')
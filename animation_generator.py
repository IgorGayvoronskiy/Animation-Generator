import math
import fbx
import tools
import metrabs_tools
import model_download
import numpy as np

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

i = input('Выберите видео: ')

frames_landmarks = metrabs_tools.get_video_data_from_json(f'json_files/video{i}.json')[0]

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

frame = 0
head_leaf_nodes = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
diff_rotated = ['rotated_hip', 'left_hip', 'right_hip', 'left_collarbone', 'right_collarbone']

# Создание FBX-сцены
manager = fbx.FbxManager.Create()
scene = fbx.FbxScene.Create(manager, 'MyScene')

# Создаем кости
skeleton_root = fbx.FbxSkeleton.Create(manager, 'root')
skeleton_root.SetSkeletonType(fbx.FbxSkeleton.EType.eRoot)

# Создаем узел скелета
root_node = fbx.FbxNode.Create(manager, 'root')
root_node.SetNodeAttribute(skeleton_root)
root_node.LclTranslation.Set(fbx.FbxDouble3(*frames_landmarks[frame]['root']))
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


def rotate_bone(name, parent_bone_node, bone_node, quat, angles):
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

    angles[name] = rotation_

    bone_node.LclRotation.Set(rotation_)


def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def build_rotation_matrix(direction, twist_reference, parent_rotation):
    """
    Строит матрицу вращения с контролем твиста:
    - direction: основное направление кости (z-ось)
    - twist_reference: вектор, задающий направление боковой оси (x-ось после ортогонализации)
    """

    direction = np.linalg.inv(parent_rotation) @ direction
    twist_reference = np.linalg.inv(parent_rotation) @ twist_reference

    z_axis = normalize(direction)

    # Убираем компоненту вдоль направления
    twist_proj = twist_reference - np.dot(twist_reference, z_axis) * z_axis
    x_axis = normalize(twist_proj)

    if np.linalg.norm(x_axis) < 1e-6:
        # Если twist_reference почти параллелен direction — берём глобальный запасной вариант
        if abs(z_axis[1]) < 0.99:
            x_axis = normalize(np.cross([0, 1, 0], z_axis))
        else:
            x_axis = normalize(np.cross([1, 0, 0], z_axis))

    y_axis = np.cross(z_axis, x_axis)

    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
    return rot_matrix


def build_rotation_matrix2(forward, up):
    forward = normalize(forward)
    right = normalize(np.cross(up, forward))
    up_corrected = normalize(np.cross(forward, right))
    rot_matrix = np.column_stack((right, up_corrected, forward))
    if np.linalg.det(rot_matrix) < 0:
        rot_matrix[:, 0] = -rot_matrix[:, 0]
    return rot_matrix


def calculate_rotation3(name, start, cur, end, standard_direction=False, frame_hip=0):
    direction = normalize(np.array(end, dtype=float) - np.array(cur, dtype=float))

    # Базовый вектор направления
    if standard_direction:
        base_direction = np.array([0, 1, 0], dtype=float)
    else:
        base_direction = normalize(np.array(cur, dtype=float) - np.array(start, dtype=float))

    # Выравниваем основное направление
    align_rot = R.align_vectors([direction], [base_direction])[0]

    up_hint = np.array([1, 0, 0])
    if name == 'hip':
        up_hint = [1, 0, 1]
    # if abs(np.dot(direction, up_hint)) > 0.9:
    #     up_hint = np.array([0, 0, 1])
    # if name == 'left_collarbone' or name == 'right_collarbone':
    #     up_hint = np.array([0, 0, 1])

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


def calculate_rotation(name, start, cur, end, standard_direction=False, frame_=0):
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
    if abs(np.dot(direction, up_hint)) >= 0.7:
        up_hint = np.array([0, 0, 1])
    if abs(np.dot(base_direction, base_hint)) >= 0.7:
        base_hint = np.array([0, 0, 1])
    if name == 'hip':
        base_hint = [1, 0, 0]
        up_hint = normalize(frames_landmarks[frame_]['left_hip'] - frames_landmarks[frame_]['hip'])
    if name == 'left_hip':
        base_hint = normalize(frames_landmarks[frame_]['left_hip'] - frames_landmarks[frame_]['hip'])
        up_hint = normalize(frames_landmarks[frame_]['left_knee'] - frames_landmarks[frame_]['left_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks[frame_]['left_foot_index'] - frames_landmarks[frame_]['left_ankle'])
    elif name == 'right_hip':
        base_hint = normalize(frames_landmarks[frame_]['left_hip'] - frames_landmarks[frame_]['hip'])
        up_hint = normalize(frames_landmarks[frame_]['right_knee'] - frames_landmarks[frame_]['right_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks[frame_]['right_foot_index'] - frames_landmarks[frame_]['right_ankle'])

    parent_matrix = nodes[bone_structure[name][0]].EvaluateGlobalTransform()
    parent_matrix = np.array([[parent_matrix.Get(i, j) for j in range(3)] for i in range(3)])
    if name in ['hip', 'left_hip', 'right_hip']:
        base_matrix = build_rotation_matrix(base_direction, base_hint, parent_matrix)
        target_matrix = build_rotation_matrix(direction, up_hint, parent_matrix)
    else:
        base_matrix = build_rotation_matrix2(base_direction, base_hint)
        target_matrix = build_rotation_matrix2(direction, up_hint)

    # Переход от base → target
    rot_matrix = target_matrix @ np.linalg.inv(base_matrix)
    return R.from_matrix(rot_matrix).as_quat()


def calculate_rotation2(name, start, cur, end, standard_direction=False, frame_=0):
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

    nodes[current] = create_bone(current, parent_point, current_point,
                                 parent_node, flag)
locations.append(cur_locations)

for current, (parent, daughter) in bone_structure.items():
    parent_node = nodes[parent]
    current_node = nodes[current]
    if daughter is None:
        if current in head_leaf_nodes:
            rotation = calculate_rotation(current, frames_landmarks[frame][parent], frames_landmarks[frame][parent],
                                          frames_landmarks[frame][current], True)
        else:
            rotation = (0, 0, 0, 0)
    else:
        if current == 'hip':
            # rotation = (0, 0, 0, 0)
            rotation = calculate_rotation(current, frames_landmarks[frame][parent], [0, 0, 0], [0, 1, 0], True, frame)
        elif current in diff_rotated:
            rotation = calculate_rotation(current, frames_landmarks[frame][parent], frames_landmarks[frame][current],
                                          frames_landmarks[frame][daughter], True)
        else:
            rotation = calculate_rotation(current, frames_landmarks[frame][parent], frames_landmarks[frame][current],
                                          frames_landmarks[frame][daughter])
    quats[current] = rotation
    rotate_bone(current, parent_node, current_node, rotation, cur_rotations)
rotations.append(cur_rotations)

for frame in tqdm(range(1, len(frames_landmarks))):
    if frame % 2 == 0:
        add_frames = [0.33, 0.66, 1]
    else:
        add_frames = [0.5, 1]
    for t in add_frames:
        hint = None
        cur_locations = {'root': frames_landmarks[frame]['root']}
        cur_rotations = {}
        for current, (parent, daughter) in bone_structure.items():
            flag = False
            current_node = nodes[current]
            bone_location = frames_landmarks[frame][current] - frames_landmarks[frame][parent]
            if current == 'hip' or current in diff_rotated or current in head_leaf_nodes:
                cur_locations[current] = bone_location
                flag = True
            locate_bone(bone_location, current_node, flag)
        locations.append(cur_locations)
        for current, (parent, daughter) in bone_structure.items():
            parent_node = nodes[parent]
            current_node = nodes[current]
            if daughter is None:
                if current in head_leaf_nodes:
                    rotation = calculate_rotation(current, frames_landmarks[frame][parent],
                                                  frames_landmarks[frame][parent],
                                                  frames_landmarks[frame][current], True)
                else:
                    rotation = (0, 0, 0, 0)
            else:
                if current == 'hip':
                    # rotation = (0, 0, 0, 0)
                    rotation = calculate_rotation(current, frames_landmarks[frame][parent],
                                                  [0, 0, 0],
                                                  [0, 1, 0], True, frame)
                elif current in diff_rotated:
                    rotation = calculate_rotation(current, frames_landmarks[frame][parent],
                                                  frames_landmarks[frame][current],
                                                  frames_landmarks[frame][daughter], True)
                else:
                    rotation = calculate_rotation(current, frames_landmarks[frame][parent],
                                                  frames_landmarks[frame][current],
                                                  frames_landmarks[frame][daughter])

            if (daughter is not None or current in head_leaf_nodes):  # тут убрал current != 'hip' and
                slerp_rotation = slerp_quaternion(quats[current], rotation, t)
                quats[current] = rotation
            else:
                slerp_rotation = rotation

            rotate_bone(current, parent_node, current_node, slerp_rotation, cur_rotations)
        rotations.append(cur_rotations)


# Сохраняем FBX
# exporter = fbx.FbxExporter.Create(manager, "")
# exporter.Initialize(f'output{i}.fbx', -1, manager.GetIOSettings())
# exporter.Export(scene)
# exporter.Destroy()
# print(f'FBX файл создан: output{i}.fbx')


def add_animation(scene, nodes, landmarks_frames, bone_struct, diff_rot, h_l_n, rotations_by_frame):
    anim_stack = fbx.FbxAnimStack.Create(scene, "Animation")
    anim_layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
    anim_stack.AddMember(anim_layer)
    scene.SetCurrentAnimationStack(anim_stack)

    for frame, landmarks in tqdm(enumerate(landmarks_frames)):
        time = fbx.FbxTime()
        time.SetFrame(frame, fbx.FbxTime.EMode.eFrames60)

        for current_bone_name, current_bone_node in nodes.items():
            x_curve = current_bone_node.LclTranslation.GetCurve(anim_layer, "X", True)
            y_curve = current_bone_node.LclTranslation.GetCurve(anim_layer, "Y", True)
            z_curve = current_bone_node.LclTranslation.GetCurve(anim_layer, "Z", True)

            x_curve.KeyModifyBegin()
            y_curve.KeyModifyBegin()
            z_curve.KeyModifyBegin()

            if current_bone_name == 'root' or current_bone_name == 'hip':
                # global_pos = landmarks[current_bone_name]
                # parent_bone_node = current_bone_node.GetParent()

                # if parent_bone_node.GetName() != 'RootNode':
                #     parent_global_pos = landmarks[parent_bone_node.GetName()]
                # else:
                #     parent_global_pos = (0, 0, 0)
                #
                # local_pos = (
                #     global_pos[0] - parent_global_pos[0],
                #     global_pos[1] - parent_global_pos[1],
                #     global_pos[2] - parent_global_pos[2]
                # )

                key_index_x = x_curve.KeyAdd(time)[0]
                key_index_y = y_curve.KeyAdd(time)[0]
                key_index_z = z_curve.KeyAdd(time)[0]

                x_curve.KeySetValue(key_index_x, landmarks_frames[frame][current_bone_name][0])
                y_curve.KeySetValue(key_index_y, landmarks_frames[frame][current_bone_name][1])
                z_curve.KeySetValue(key_index_z, landmarks_frames[frame][current_bone_name][2])

                x_curve.KeySetInterpolation(key_index_x, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant)
                y_curve.KeySetInterpolation(key_index_y, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant)
                z_curve.KeySetInterpolation(key_index_z, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant)

            x_curve.KeyModifyEnd()
            y_curve.KeyModifyEnd()
            z_curve.KeyModifyEnd()

            x_rot_curve = current_bone_node.LclRotation.GetCurve(anim_layer, "X", True)
            y_rot_curve = current_bone_node.LclRotation.GetCurve(anim_layer, "Y", True)
            z_rot_curve = current_bone_node.LclRotation.GetCurve(anim_layer, "Z", True)

            x_rot_curve.KeyModifyBegin()
            y_rot_curve.KeyModifyBegin()
            z_rot_curve.KeyModifyBegin()

            if (current_bone_name != 'root' and  # тут убрал: and current_bone_name != 'hip'
                    current_bone_name != 'head_top' and current_bone_name not in h_l_n):
                key_index_x = x_rot_curve.KeyAdd(time)[0]
                key_index_y = y_rot_curve.KeyAdd(time)[0]
                key_index_z = z_rot_curve.KeyAdd(time)[0]

                x_rot_curve.KeySetValue(key_index_x, rotations_by_frame[frame][current_bone_name][0])
                y_rot_curve.KeySetValue(key_index_y, rotations_by_frame[frame][current_bone_name][1])
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


# from collections import defaultdict
# import numpy as np
#
# # Структура: bone_name -> channel -> [angle_frame0, angle_frame1, ...]
# euler_angles = defaultdict(lambda: defaultdict(list))
#
# for frame in range(len(rotations)):
#     for bone_name in bone_structure:
#         if bone_name in ['root', 'hip']:
#             continue
#         angles = rotations[frame][bone_name]  # (x, y, z)
#         euler_angles[bone_name]['x'].append(angles[0])
#         euler_angles[bone_name]['y'].append(angles[1])
#         euler_angles[bone_name]['z'].append(angles[2])
#
# for bone_name in bone_structure:
#     if bone_name in ['root', 'hip']:
#         continue
#
#     for axis in ['x', 'y', 'z']:
#         raw = np.radians(euler_angles[bone_name][axis])
#         #unwrapped = np.unwrap(raw, discont=np.radians(180))
#         euler_angles[bone_name][axis] = np.degrees(raw)

scene2, nodes2 = model_download.get_skeleton_nodes('models/spider_man_model3.fbx')

add_animation(scene2, nodes2, locations, bone_structure, diff_rotated, head_leaf_nodes, rotations)

# Экспорт анимированного FBX
exporter = fbx.FbxExporter.Create(manager, "")
exporter.Initialize(f"animated_model_output{i}.fbx", -1, manager.GetIOSettings())
exporter.Export(scene2)
exporter.Destroy()
print(f"FBX анимация создана: animated_model_output{i}.fbx")

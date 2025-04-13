import math

import fbx
import tools
import metrabs_tools
import numpy as np

from scipy.spatial.transform import Rotation as R

i = input('Выберите видео: ')

frames_landmarks = metrabs_tools.get_data_from_json(f'json_files/video{i}.json')[0]

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
list_group1 = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
diff_rotated = ['rotated_hip', 'left_hip', 'right_hip', 'left_collarbone', 'right_collarbone', 'had_center']

# Создание FBX-сцены
manager = fbx.FbxManager.Create()
scene = fbx.FbxScene.Create(manager, 'MyScene')

# Создаем кости
skeleton_root = fbx.FbxSkeleton.Create(manager, 'root')
skeleton_root.SetSkeletonType(fbx.FbxSkeleton.EType.eRoot)

# Создаем узел скелета
root_node = fbx.FbxNode.Create(manager, 'root')
root_node.SetNodeAttribute(skeleton_root)
root_node.LclTranslation.Set(fbx.FbxDouble3(*frames_landmarks[0]['root']))
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


def rotate_bone(name, parent_bone_node, bone_node, quat):

    # if name == 'left_hip' or name == 'left_knee' or name == 'left_ankle' or name == 'left_heel' or name == 'left_foot_index'\
    #         or name == 'right_hip' or name == 'right_knee':
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


# Добавляем кости в сцену
nodes = {'root': root_node}

for current, (parent, daughter) in bone_structure.items():
    flag = False
    parent_node = nodes[parent]
    current_point = frames_landmarks[0][current]
    parent_point = frames_landmarks[0][parent]
    if current == 'hip' or current in diff_rotated or current in list_group1:
        flag = True

    nodes[current] = create_bone(current, parent_point, current_point,
                                 parent_node, True)

# for current, (parent, daughter) in bone_structure.items():
#     parent_node = nodes[parent]
#     current_node = nodes[current]
#     if daughter is None:
#         if current in list_group1:
#             rotation = calculate_rotation(frames_landmarks[0][parent], frames_landmarks[0][parent],
#                                           frames_landmarks[0][current], True)
#         else:
#             rotation = (0, 0, 0, 0)
#     else:
#         if current == 'hip':
#             rotation = (0, 0, 0, 0)
#         elif current in diff_rotated:
#             rotation = calculate_rotation(frames_landmarks[0][parent], frames_landmarks[0][current],
#                                           frames_landmarks[0][daughter], True)
#         else:
#             rotation = calculate_rotation(frames_landmarks[0][parent], frames_landmarks[0][current],
#                                           frames_landmarks[0][daughter])
#
#     rotate_bone(current, parent_node, current_node, rotation)

# Сохраняем FBX
# exporter = fbx.FbxExporter.Create(manager, "")
# exporter.Initialize(f'output{i}.fbx', -1, manager.GetIOSettings())
# exporter.Export(scene)
# exporter.Destroy()
# print(f'FBX файл создан: output{i}.fbx')


def add_animation(scene, nodes, landmarks_frames):
    anim_stack = fbx.FbxAnimStack.Create(scene, "Animation")
    anim_layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
    anim_stack.AddMember(anim_layer)
    scene.SetCurrentAnimationStack(anim_stack)

    for frame, landmarks in enumerate(landmarks_frames):
        time = fbx.FbxTime()
        time.SetFrame(frame, fbx.FbxTime.EMode.eFrames24)

        for bone_name, node in nodes.items():
            x_curve = node.LclTranslation.GetCurve(anim_layer, "X", True)
            y_curve = node.LclTranslation.GetCurve(anim_layer, "Y", True)
            z_curve = node.LclTranslation.GetCurve(anim_layer, "Z", True)

            x_curve.KeyModifyBegin()
            y_curve.KeyModifyBegin()
            z_curve.KeyModifyBegin()

            if bone_name in landmarks:
                if bone_name != 'key':
                    global_pos = landmarks[bone_name]
                else:
                    global_pos = (0, 0, 0)
                parent_node = node.GetParent()

                if parent_node.GetName() != 'RootNode' and parent_node.GetName() != 'cvvcvcvc':
                    parent_global_pos = landmarks[parent_node.GetName()]
                else:
                    parent_global_pos = (0, 0, 0)

                local_pos = (
                    global_pos[0] - parent_global_pos[0],
                    global_pos[1] - parent_global_pos[1],
                    global_pos[2] - parent_global_pos[2]
                )

                key_index_x = x_curve.KeyAdd(time)[0]
                key_index_y = y_curve.KeyAdd(time)[0]
                key_index_z = z_curve.KeyAdd(time)[0]

                x_curve.KeySetValue(key_index_x, local_pos[0])
                y_curve.KeySetValue(key_index_y, local_pos[1])
                z_curve.KeySetValue(key_index_z, local_pos[2])

                x_curve.KeySetInterpolation(key_index_x, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
                y_curve.KeySetInterpolation(key_index_y, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
                z_curve.KeySetInterpolation(key_index_z, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)

            x_curve.KeyModifyEnd()
            y_curve.KeyModifyEnd()
            z_curve.KeyModifyEnd()


add_animation(scene, nodes, frames_landmarks)

# Экспорт анимированного FBX
exporter = fbx.FbxExporter.Create(manager, "")
exporter.Initialize(f"animated_output{i}.fbx", -1, manager.GetIOSettings())
exporter.Export(scene)
exporter.Destroy()
print(f"FBX анимация создана: animated_output{i}.fbx")

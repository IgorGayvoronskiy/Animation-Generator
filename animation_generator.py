import shutil
import math
import fbx
import metrabs_tools
import model_download
import numpy as np
import time as tm

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


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

    # new_local = add_twist(name, new_local)

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
    return transformed[:3]  # вернём обратно в 3D


def twist_rotate(name, parent_bone_node, bone_node, frames_landmarks, direction, angles, frame_=0):
    direction = normalize(direction)

    if name == 'hip':
        up_hint = normalize(frames_landmarks[frame_]['left_hip'] - frames_landmarks[frame_]['hip'])
        needed_v = normalize(up_hint)
    elif name == 'rotated_left_hip':
        up_hint = normalize(frames_landmarks[frame_]['left_knee'] - frames_landmarks[frame_]['left_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks[frame_]['left_ankle'] - frames_landmarks[frame_]['left_foot_index'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_hip':
        up_hint = normalize(frames_landmarks[frame_]['right_knee'] - frames_landmarks[frame_]['right_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks[frame_]['right_ankle'] - frames_landmarks[frame_]['right_foot_index'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_knee':
        up_hint = normalize(frames_landmarks[frame_]['left_ankle'] - frames_landmarks[frame_]['left_foot_index'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks[frame_]['left_knee'] - frames_landmarks[frame_]['left_hip'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_knee':
        up_hint = normalize(frames_landmarks[frame_]['right_ankle'] - frames_landmarks[frame_]['right_foot_index'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks[frame_]['right_knee'] - frames_landmarks[frame_]['right_hip'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_ankle':
        up_hint = normalize(frames_landmarks[frame_]['left_knee'] - frames_landmarks[frame_]['left_ankle'])
        # if abs(np.dot(direction, up_hint)) >= 0.99:
        #     up_hint = normalize(frames_landmarks[frame_]['left_knee'] - frames_landmarks[frame_]['left_hip'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_ankle':
        up_hint = normalize(frames_landmarks[frame_]['right_knee'] - frames_landmarks[frame_]['right_ankle'])
        # if abs(np.dot(direction, up_hint)) >= 0.99:
        #     up_hint = normalize(frames_landmarks[frame_]['right_knee'] - frames_landmarks[frame_]['right_hip'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_left_collarbone':
        up_hint = normalize(frames_landmarks[frame_]['neck'] - frames_landmarks[frame_]['thor'])
        # if abs(np.dot(direction, up_hint)) >= 0.99:
        #     up_hint = normalize(frames_landmarks[frame_]['left_knee'] - frames_landmarks[frame_]['left_hip'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_collarbone':
        up_hint = normalize(frames_landmarks[frame_]['neck'] - frames_landmarks[frame_]['thor'])
        # if abs(np.dot(direction, up_hint)) >= 0.99:
        #     up_hint = normalize(frames_landmarks[frame_]['right_knee'] - frames_landmarks[frame_]['right_hip'])
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
        up_hint = normalize(frames_landmarks[frame_]['left_thumb'] - frames_landmarks[frame_]['left_wrist'])
        # if abs(np.dot(direction, up_hint)) >= 0.99:
        #     up_hint = normalize(frames_landmarks[frame_]['left_knee'] - frames_landmarks[frame_]['left_hip'])
        needed_v = np.cross(direction, up_hint)
        needed_v = normalize(needed_v)
    elif name == 'rotated_right_elbow':
        up_hint = normalize(frames_landmarks[frame_]['right_thumb'] - frames_landmarks[frame_]['right_wrist'])
        # if abs(np.dot(direction, up_hint)) >= 0.99:
        #     up_hint = normalize(frames_landmarks[frame_]['right_knee'] - frames_landmarks[frame_]['right_hip'])
        needed_v = normalize(up_hint)
    elif name == 'rotated_left_wrist':
        up_hint = normalize(frames_landmarks[frame_]['left_thumb'] - frames_landmarks[frame_]['left_wrist'])
        needed_v = normalize(up_hint)
        # if abs(np.dot(direction, up_hint)) >= 0.99:
        #     up_hint = normalize(frames_landmarks[frame_]['left_knee'] - frames_landmarks[frame_]['left_hip'])
    elif name == 'rotated_right_wrist':
        up_hint = normalize(frames_landmarks[frame_]['right_thumb'] - frames_landmarks[frame_]['right_wrist'])
        needed_v = normalize(up_hint)
        # if abs(np.dot(direction, up_hint)) >= 0.99:
        #     up_hint = normalize(frames_landmarks[frame_]['right_knee'] - frames_landmarks[frame_]['right_hip'])

    if name != 'hip':
        parent_global = parent_bone_node.EvaluateGlobalTransform()
        parent_global_inverse = parent_global.Inverse()

        local_needed_v = normalize(transform_vector(parent_global_inverse, needed_v))
        local_direction = np.array([0, 1, 0])

        quat = quaternion_twist(np.array([1, 0, 0]), local_needed_v, local_direction).as_quat()

        # quat_fbx = fbx.FbxQuaternion(quat[0], quat[1], quat[2], quat[3])
        # rotation_matrix = fbx.FbxAMatrix()
        # rotation_matrix.SetQ(quat_fbx)
        #
        # rotation_ = rotation_matrix.GetR()
        #
        # rotation_ = fbx.FbxDouble3(rotation_[0], rotation_[1], rotation_[2])
        #
        # angles[name] = rotation_
        #
        # bone_node.SetRotationOrder(fbx.FbxNode.EPivotSet.eSourcePivot, fbx.EFbxRotationOrder.eEulerYXZ)
        # bone_node.LclRotation.Set(rotation_)

        return quat
    else:
        parent_global = parent_bone_node.EvaluateGlobalTransform()
        parent_global_inverse = parent_global.Inverse()

        local_needed_v = normalize(transform_vector(parent_global_inverse, needed_v))
        local_direction = np.array([0, 1, 0])

        quat = quaternion_twist(np.array([1, 0, 0]), local_needed_v, local_direction).as_quat()

        return quat


def add_twist(name, rot_matr, twist_axis=None):
    if twist_axis is None:
        twist_axis = np.array([0, 1, 0])

    q = rot_matr.GetQ()
    numpy_q = np.array([q[0], q[1], q[2], q[3]], dtype=float)
    w, v = numpy_q[-1], numpy_q[:-1]

    proj = np.dot(v, twist_axis) * twist_axis
    twist_q = np.array([proj[0], proj[1], proj[2], w])
    twist_q = normalize(twist_q)
    twist_len = np.linalg.norm(twist_q)
    if twist_len <= 1e-8:
        twist_q = np.array([0.0, 0.0, 0.0, 1.0])

    twist_q_fbx = fbx.FbxQuaternion(*twist_q)
    twist_matr = fbx.FbxAMatrix()
    twist_matr.SetQ(twist_q_fbx)

    swing_matr = rot_matr * twist_matr.Inverse()

    a = swing_matr.GetQ()
    swing_numpy = np.array([a[0], a[1], a[2], a[3]])
    swing_numpy = normalize(swing_numpy)

    swing_q = R.from_quat(swing_numpy)
    # swing_matr = fbx.FbxAMatrix()
    # swing_matr.SetQ(swing_q)

    rot = R.from_quat(numpy_q)
    rotated_twist_axis = rot.apply(twist_axis)

    proj = np.dot(v, rotated_twist_axis) * rotated_twist_axis
    new_twist_q = np.array([0, 100, 0, w])
    new_twist_q = normalize(new_twist_q)

    # new_twist_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    # new_twist_q = normalize(new_twist_q)
    new_twist_q_fbx = fbx.FbxQuaternion(*new_twist_q)
    new_twist_matr = fbx.FbxAMatrix()
    new_twist_matr.SetQ(new_twist_q_fbx)

    # initial_side = np.array([1, 0, 0])
    # rotated_side = swing_q.apply(initial_side)
    #
    # desired = normalize(np.array([1, 0, 0]))
    # rotated_side = normalize(rotated_side)
    #
    # # Угол между текущей боковой и желаемой
    # twist_axis = swing_q.apply(twist_axis)
    # axis = normalize(twist_axis)
    # dot = np.clip(np.dot(rotated_side, desired), -1.0, 1.0)
    # angle = np.arccos(dot)
    #
    # # Направление вращения
    # cross = np.cross(rotated_side, desired)
    # sign = np.sign(np.dot(cross, axis))
    # angle *= sign
    #
    # twist_rot = R.from_rotvec(axis * angle).as_quat()
    #
    # twist_q = fbx.FbxQuaternion(twist_rot[0], twist_rot[1], twist_rot[2], twist_rot[3])
    #
    # twist_matr = fbx.FbxAMatrix()
    # twist_matr.SetQ(twist_q)

    if name == 'left_hip':
        return swing_matr * twist_matr
    else:
        return swing_matr


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


def calculate_rotation2(frames_landmarks, nodes, bone_structure, name,
                        start, cur, end, standard_direction=False, frame_=0):
    direction = np.array(end, dtype=float) - np.array(cur, dtype=float)
    direction = direction / np.linalg.norm(direction)

    if standard_direction:
        base_direction = np.array([0, 1, 0], dtype=float)
    else:
        base_direction = np.array(cur, dtype=float) - np.array(start, dtype=float)
        base_direction = base_direction / np.linalg.norm(base_direction)

    rotation_ = R.align_vectors([direction], [base_direction])[0]
    quat = rotation_.as_quat()
    quat_np = np.array([quat[0], quat[1], quat[2], quat[3]], dtype=float)
    quat = R.from_quat(quat_np)

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
        # up_hint = quat.apply(base_hint)
        up_hint = normalize(frames_landmarks[frame_]['left_knee'] - frames_landmarks[frame_]['left_ankle'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks[frame_]['left_foot_index'] - frames_landmarks[frame_]['left_ankle'])
    elif name == 'right_hip':
        # base_hint = normalize(frames_landmarks[frame_]['right_hip'] - frames_landmarks[frame_]['hip'])
        # up_hint = normalize(frames_landmarks[frame_]['right_knee'] - frames_landmarks[frame_]['right_ankle'])
        base_hint = np.array([1, 0, 0])
        up_hint = normalize(frames_landmarks[frame_]['right_hip'] - frames_landmarks[frame_]['hip'])
        if abs(np.dot(direction, up_hint)) >= 0.99:
            up_hint = normalize(frames_landmarks[frame_]['right_foot_index'] - frames_landmarks[frame_]['right_ankle'])

    parent_matrix = nodes[bone_structure[name][0]].EvaluateGlobalTransform()
    parent_matrix = np.array([[parent_matrix.Get(i, j) for j in range(3)] for i in range(3)])
    if name in ['hip', 'left_hip', 'right_hip']:
        base_matrix = build_rotation_matrix2(base_direction, base_hint)
        target_matrix = build_rotation_matrix2(direction, up_hint)
    else:
        base_matrix = build_rotation_matrix2(base_direction, base_hint)
        target_matrix = build_rotation_matrix2(direction, up_hint)

    rot_matrix = target_matrix @ np.linalg.inv(base_matrix)
    return R.from_matrix(rot_matrix).as_quat()


def calculate_rotation(frames_landmarks, nodes, bone_structure, name,
                       start, cur, end, standard_direction=False, frame_=0):
    direction = np.array(end, dtype=float) - np.array(cur, dtype=float)
    direction = direction / np.linalg.norm(direction)

    if standard_direction:
        base_direction = np.array([0, 1, 0], dtype=float)
    else:
        base_direction = np.array(cur, dtype=float) - np.array(start, dtype=float)
        base_direction = base_direction / np.linalg.norm(base_direction)

    rotation_ = R.align_vectors([direction], [base_direction])[0]
    quat = rotation_.as_quat()
    return quat


def add_animation(scene, nodes, landmarks_frames, bone_struct, diff_rot, h_l_n, rotations_by_frame,
                  progress_callback=None):
    anim_stack = fbx.FbxAnimStack.Create(scene, "Animation")
    anim_layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
    anim_stack.AddMember(anim_layer)
    scene.SetCurrentAnimationStack(anim_stack)

    total = len(landmarks_frames)

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


def create_animation(key_points_data_path, model_path, from_json=True, progress_callback=None):
    if from_json:
        frames_landmarks = metrabs_tools.get_video_data_from_json(key_points_data_path)[0]
    else:
        frames_landmarks = metrabs_tools.get_data_from_video(key_points_data_path, progress_callback)[0]

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
                rotation = calculate_rotation(frames_landmarks, nodes, bone_structure, current,
                                              frames_landmarks[frame][parent], frames_landmarks[frame][parent],
                                              frames_landmarks[frame][current], True)
            else:
                rotation = np.array([0, 0, 0, 1], dtype=float)
        else:
            if current == 'hip':
                rotation = calculate_rotation(frames_landmarks, nodes, bone_structure, current,
                                              frames_landmarks[frame][parent],
                                              (frames_landmarks[frame]['left_hip'] +
                                               frames_landmarks[frame]['right_hip']) / 2,
                                              frames_landmarks[frame][current], True)

            elif current in diff_rotated:
                rotation = calculate_rotation(frames_landmarks, nodes, bone_structure, current,
                                              frames_landmarks[frame][parent], frames_landmarks[frame][current],
                                              frames_landmarks[frame][daughter], True)
            else:
                rotation = calculate_rotation(frames_landmarks, nodes, bone_structure, current,
                                              frames_landmarks[frame][parent], frames_landmarks[frame][current],
                                              frames_landmarks[frame][daughter])

        twist = np.array([0, 0, 0, 1], dtype=float)
        # if current.startswith('rotated_') and current != 'rotated_hip' and bone_structure[parent][1] is not None:
        if ((current.startswith('rotated_') and current not in ['rotated_hip', 'rotated_left_han', 'rotated_right_han',
                                                                'rotated_left_foot_index', 'rotated_right_foot_index'])
                or current == 'hip'):
            if current == 'hip':
                direction = frames_landmarks[0][current] - (frames_landmarks[0]['left_hip'] +
                                                            frames_landmarks[0]['right_hip']) / 2
            else:
                direction = frames_landmarks[0][bone_structure[parent][1]] - frames_landmarks[0][parent]
            twist = twist_rotate(current, parent_node, current_node, frames_landmarks, direction, cur_rotations)

        rotate_bone(current, parent_node, current_node, rotation, twist, cur_rotations)

        quats[current] = rotation

    rotations.append(cur_rotations)

    total = len(frames_landmarks)

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
                        rotation = calculate_rotation(frames_landmarks, nodes, bone_structure, current,
                                                      frames_landmarks[frame][parent], frames_landmarks[frame][parent],
                                                      frames_landmarks[frame][current], True, frame)
                    else:
                        rotation = np.array([0, 0, 0, 1], dtype=float)
                else:
                    if current == 'hip':
                        # rotation = (0, 0, 0, 0)
                        rotation = calculate_rotation(frames_landmarks, nodes, bone_structure, current,
                                                      frames_landmarks[frame][parent],
                                                      (frames_landmarks[frame]['left_hip'] +
                                                       frames_landmarks[frame]['right_hip']) / 2,
                                                      frames_landmarks[frame][current], True, frame)

                    elif current in diff_rotated:
                        rotation = calculate_rotation(frames_landmarks, nodes, bone_structure, current,
                                                      frames_landmarks[frame][parent], frames_landmarks[frame][current],
                                                      frames_landmarks[frame][daughter], True, frame)
                    else:
                        rotation = calculate_rotation(frames_landmarks, nodes, bone_structure, current,
                                                      frames_landmarks[frame][parent], frames_landmarks[frame][current],
                                                      frames_landmarks[frame][daughter], False, frame)

                if (((daughter is not None or current in head_leaf_nodes)
                     and not current.startswith(
                            'rotated_')) or current == 'rotated_hip'):  # тут убрал current != 'hip' and
                    slerp_rotation = slerp_quaternion(quats[current], rotation, t)
                    quats[current] = rotation
                else:
                    slerp_rotation = rotation

                twist = np.array([0, 0, 0, 1], dtype=float)

                # if current.startswith('rotated_') and current != 'rotated_hip' and bone_structure[parent][1] is not None:
                if ((current.startswith('rotated_') and current not in ['rotated_hip', 'rotated_left_han',
                                                                        'rotated_right_han', 'rotated_left_foot_index',
                                                                        'rotated_right_foot_index'])
                        or current == 'hip'):
                    if current == 'hip':
                        direction = frames_landmarks[frame][current] - (frames_landmarks[frame]['left_hip'] +
                                                                        frames_landmarks[frame]['right_hip']) / 2
                    else:
                        direction = frames_landmarks[frame][bone_structure[parent][1]] - frames_landmarks[frame][parent]
                    twist = twist_rotate(current, parent_node, current_node, frames_landmarks, direction, cur_rotations,
                                         frame)

                rotate_bone(current, parent_node, current_node, slerp_rotation, twist, cur_rotations)
            rotations.append(cur_rotations)

            if progress_callback:
                progress_callback(frame, total, 'Обработка данных')

    scene2, nodes2 = model_download.get_skeleton_nodes(model_path)
    add_animation(scene2, nodes2, locations, bone_structure, diff_rotated, head_leaf_nodes, rotations,
                  progress_callback)

    # Экспорт анимированного FBX
    print(model_path)
    model_name = model_path.split('/')[-1].split('.')[0]
    video_name = key_points_data_path.split('/')[-1].split('.')[0]
    exporter = fbx.FbxExporter.Create(manager, "")
    exporter.Initialize(f"animated_models/animated_{model_name}_from_{video_name}.fbx", -1, manager.GetIOSettings())
    exporter.Export(scene2)
    exporter.Destroy()
    print(f"FBX анимация создана: animated_{model_name}_from_{video_name}.fbx")

    if not from_json:
        source_path = key_points_data_path
        destination_path = f"animated_models/animated_{model_name}_from_{video_name}.mp4"
        shutil.copy2(source_path, destination_path)


if __name__ == '__main__':
    input_type = input('Введи v если видео или j, если json: ')
    if input_type == 'v':
        i = input('Введите номер тестового видео: ')
        create_animation(f'Source/videos/video{i}.mp4',
                         'Source/rigged_models/spider_man_model_B.fbx', False)
    elif input_type == 'j':
        i = input('Введите номер тестового json: ')
        create_animation(f'Source/json_files/video{i}.json',
                         'Source/rigged_models/spider_man_model_B.fbx', True)

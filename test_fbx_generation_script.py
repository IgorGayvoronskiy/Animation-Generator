import fbx
import tools
import numpy as np


# Пример 3D-ключевых точек (замени своими данными)
keypoints_3d = {
    "hips": np.array([0, 0, 0]),  # Центр таза (нулевая точка)
    "spine": np.array([0, 10, 0]),  # Позвоночник выше таза
    "chest": np.array([0, 20, 0]),  # Грудь
    "neck": np.array([0, 30, 0]),  # Шея
    "head": np.array([0, 35, 0]),  # Голова

    # Руки вытянуты в стороны (ось X)
    "left_shoulder": np.array([-10, 25, 0]),
    "left_elbow": np.array([-20, 25, 0]),
    "left_wrist": np.array([-30, 25, 0]),

    "right_shoulder": np.array([10, 25, 0]),
    "right_elbow": np.array([20, 25, 0]),
    "right_wrist": np.array([30, 25, 0]),

    # Ноги стоят прямо (ось Y вниз)
    "left_hip": np.array([-5, 0, 0]),
    "left_knee": np.array([-5, -15, 0]),
    "left_ankle": np.array([-5, -30, 0]),

    "right_hip": np.array([5, 0, 0]),
    "right_knee": np.array([5, -15, 0]),
    "right_ankle": np.array([5, -30, 0]),
}


bones = [
    ("hips", "spine"), ("spine", "chest"), ("chest", "neck"), ("neck", "head"),
    ("chest", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("chest", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("hips", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("hips", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

# Создание FBX-сцены
manager = fbx.FbxManager.Create()
scene = fbx.FbxScene.Create(manager, "MyScene")

# Создаем кости
skeleton_root = fbx.FbxSkeleton.Create(manager, "Root")
skeleton_root.SetSkeletonType(fbx.FbxSkeleton.EType.eRoot)

# Создаем узел скелета
root_node = fbx.FbxNode.Create(manager, "RootNode")
root_node.SetNodeAttribute(skeleton_root)
scene.GetRootNode().AddChild(root_node)


# Функция создания кости
def create_bone(name, start, end, parent_node):
    bone = fbx.FbxSkeleton.Create(manager, name)
    bone.SetSkeletonType(fbx.FbxSkeleton.EType.eLimbNode)

    bone_node = fbx.FbxNode.Create(manager, name)
    bone_node.SetNodeAttribute(bone)

    bone_node.LclTranslation.Set(fbx.FbxDouble3(*start))
    parent_node.AddChild(bone_node)
    return bone_node


# Добавляем кости в сцену
nodes = {"hips": root_node}
for (start, end) in bones:
    parent_node = nodes[start]
    nodes[end] = create_bone(end, keypoints_3d[start], keypoints_3d[end], parent_node)

# Сохраняем FBX
exporter = fbx.FbxExporter.Create(manager, "")
# exporter.Initialize("output.fbx", -1, manager.GetIOSettings())
# exporter.Export(scene)
# exporter.Destroy()
# print("FBX файл создан: output.fbx")


def add_animation(scene, nodes, keypoints_frames):
    anim_stack = fbx.FbxAnimStack.Create(scene, "Animation")
    anim_layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
    anim_stack.AddMember(anim_layer)
    scene.SetCurrentAnimationStack(anim_stack)

    # Для каждого узла создаем анимационные кривые
    for bone_name, node in nodes.items():
        x_curve = node.LclTranslation.GetCurve(anim_layer, "X", True)
        y_curve = node.LclTranslation.GetCurve(anim_layer, "Y", True)
        z_curve = node.LclTranslation.GetCurve(anim_layer, "Z", True)

        x_curve.KeyModifyBegin()
        y_curve.KeyModifyBegin()
        z_curve.KeyModifyBegin()

        for frame, keypoints in enumerate(keypoints_frames):
            time = fbx.FbxTime()
            time.SetFrame(frame, fbx.FbxTime.EMode.eFrames24)

            if bone_name in keypoints:
                pos = keypoints[bone_name]

                key_index_x = x_curve.KeyAdd(time)[0]
                key_index_y = y_curve.KeyAdd(time)[0]
                key_index_z = z_curve.KeyAdd(time)[0]

                x_curve.KeySetValue(key_index_x, pos[0])
                y_curve.KeySetValue(key_index_y, pos[1])
                z_curve.KeySetValue(key_index_z, pos[2])

                # Добавляем линейную интерполяцию между кадрами
                x_curve.KeySetInterpolation(key_index_x, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
                y_curve.KeySetInterpolation(key_index_y, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
                z_curve.KeySetInterpolation(key_index_z, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)

        x_curve.KeyModifyEnd()
        y_curve.KeyModifyEnd()
        z_curve.KeyModifyEnd()


# Пример кадров (замени на реальные данные)
frames = [
    # Кадр 1: Исходное положение (руки вытянуты в стороны)
    {
        "hips": np.array([0, 0, 0]),  # Центр таза (нулевая точка)
        "spine": np.array([0, 10, 0]),  # Позвоночник выше таза
        "chest": np.array([0, 20, 0]),  # Грудь
        "neck": np.array([0, 30, 0]),  # Шея
        "head": np.array([0, 35, 0]),  # Голова

        "left_shoulder": np.array([-10, 25, 0]),
        "left_elbow": np.array([-20, 25, 0]),
        "left_wrist": np.array([-30, 25, 0]),
        "right_shoulder": np.array([10, 25, 0]),
        "right_elbow": np.array([20, 25, 0]),
        "right_wrist": np.array([30, 25, 0]),

        "left_hip": np.array([-5, 0, 0]),
        "left_knee": np.array([-5, -15, 0]),
        "left_ankle": np.array([-5, -30, 0]),
        "right_hip": np.array([5, 0, 0]),
        "right_knee": np.array([5, -15, 0]),
        "right_ankle": np.array([5, -30, 0]),
    },

    # Кадр 2: Руки поднимаются (ось Y)
    {
        "hips": np.array([0, 0, 0]),  # Центр таза (нулевая точка)
        "spine": np.array([0, 10, 0]),  # Позвоночник выше таза
        "chest": np.array([0, 20, 0]),  # Грудь
        "neck": np.array([0, 30, 0]),  # Шея
        "head": np.array([0, 35, 0]),  # Голова

        "left_shoulder": np.array([-10, 30, 0]),
        "left_elbow": np.array([-15, 35, 0]),
        "left_wrist": np.array([-20, 40, 0]),
        "right_shoulder": np.array([10, 30, 0]),
        "right_elbow": np.array([15, 35, 0]),
        "right_wrist": np.array([20, 40, 0]),

        "left_hip": np.array([-5, 0, 0]),
        "left_knee": np.array([-5, -15, 0]),
        "left_ankle": np.array([-5, -30, 0]),
        "right_hip": np.array([5, 0, 0]),
        "right_knee": np.array([5, -15, 0]),
        "right_ankle": np.array([5, -30, 0]),
    },

    # Кадр 3: Руки подняты вверх (максимальная точка)
    {
        "hips": np.array([0, 0, 0]),  # Центр таза (нулевая точка)
        "spine": np.array([0, 10, 0]),  # Позвоночник выше таза
        "chest": np.array([0, 20, 0]),  # Грудь
        "neck": np.array([0, 30, 0]),  # Шея
        "head": np.array([0, 35, 0]),  # Голова

        "left_shoulder": np.array([-10, 35, 0]),
        "left_elbow": np.array([-10, 40, 0]),
        "left_wrist": np.array([-10, 45, 0]),
        "right_shoulder": np.array([10, 35, 0]),
        "right_elbow": np.array([10, 40, 0]),
        "right_wrist": np.array([10, 45, 0]),

        "left_hip": np.array([-5, 0, 0]),
        "left_knee": np.array([-5, -15, 0]),
        "left_ankle": np.array([-5, -30, 0]),
        "right_hip": np.array([5, 0, 0]),
        "right_knee": np.array([5, -15, 0]),
        "right_ankle": np.array([5, -30, 0]),
    },

    # Кадр 4: Руки опускаются обратно (ось Y уменьшается)
    {
        "hips": np.array([0, 0, 0]),  # Центр таза (нулевая точка)
        "spine": np.array([0, 10, 0]),  # Позвоночник выше таза
        "chest": np.array([0, 20, 0]),  # Грудь
        "neck": np.array([0, 30, 0]),  # Шея
        "head": np.array([0, 35, 0]),  # Голова

        "left_shoulder": np.array([-10, 30, 0]),
        "left_elbow": np.array([-15, 35, 0]),
        "left_wrist": np.array([-20, 40, 0]),
        "right_shoulder": np.array([10, 30, 0]),
        "right_elbow": np.array([15, 35, 0]),
        "right_wrist": np.array([20, 40, 0]),

        "left_hip": np.array([-5, 0, 0]),
        "left_knee": np.array([-5, -15, 0]),
        "left_ankle": np.array([-5, -30, 0]),
        "right_hip": np.array([5, 0, 0]),
        "right_knee": np.array([5, -15, 0]),
        "right_ankle": np.array([5, -30, 0]),
    },

    # Кадр 5: Возвращение в исходное положение
    {
        "hips": np.array([0, 0, 0]),  # Центр таза (нулевая точка)
        "spine": np.array([0, 10, 0]),  # Позвоночник выше таза
        "chest": np.array([0, 20, 0]),  # Грудь
        "neck": np.array([0, 30, 0]),  # Шея
        "head": np.array([0, 35, 0]),  # Голова

        "left_shoulder": np.array([-10, 25, 0]),
        "left_elbow": np.array([-20, 25, 0]),
        "left_wrist": np.array([-30, 25, 0]),
        "right_shoulder": np.array([10, 25, 0]),
        "right_elbow": np.array([20, 25, 0]),
        "right_wrist": np.array([30, 25, 0]),

        "left_hip": np.array([-5, 0, 0]),
        "left_knee": np.array([-5, -15, 0]),
        "left_ankle": np.array([-5, -30, 0]),
        "right_hip": np.array([5, 0, 0]),
        "right_knee": np.array([5, -15, 0]),
        "right_ankle": np.array([5, -30, 0]),
    },
]


add_animation(scene, nodes, frames)

# Экспорт анимированного FBX
exporter.Initialize("animated_output.fbx", -1, manager.GetIOSettings())
exporter.Export(scene)
exporter.Destroy()
print("FBX анимация создана: animated_output.fbx")


import fbx


# Создаём FBX-менеджер
manager = fbx.FbxManager.Create()


# Функция для загрузки FBX-файла
def load_fbx_scene(filename):
    importer = fbx.FbxImporter.Create(manager, "")
    scene = fbx.FbxScene.Create(manager, filename)
    importer.Initialize(filename, -1, manager.GetIOSettings())
    importer.Import(scene)
    importer.Destroy()
    return scene


# Загружаем модель
scene_model = load_fbx_scene("models/78237.fbx")


# Функция для поиска мэша (3D-модели)
def find_mesh(node):
    if node.GetNodeAttribute() and node.GetNodeAttribute().GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
        return node
    for i in range(node.GetChildCount()):
        mesh = find_mesh(node.GetChild(i))
        if mesh:
            return mesh
    return None


# Находим модель в сцене
mesh_node = find_mesh(scene_model.GetRootNode())

if not mesh_node:
    print("❌ Ошибка: Модель не найдена!")
else:
    print("✅ Найдена модель:", mesh_node.GetName())

    # Применяем изменения к модели
    mesh_node.LclScaling.Set(fbx.FbxDouble3(20, 20, 20))   # Увеличиваем в 100 раз
    mesh_node.LclTranslation.Set(fbx.FbxDouble3(0, 0, 0))     # Сдвигаем (подставь нужные координаты)

    # Экспортируем новую модель
    exporter = fbx.FbxExporter.Create(manager, "")
    exporter.Initialize("output_model.fbx", -1, manager.GetIOSettings())
    exporter.Export(scene_model)  # Экспортируем сцену с обновлённой моделью
    exporter.Destroy()

    print("✅ Экспорт завершён! Файл сохранён как output_model.fbx")

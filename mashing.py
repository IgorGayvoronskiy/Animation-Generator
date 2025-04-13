import fbx

# Создаём менеджер FBX
manager = fbx.FbxManager.Create()


# Функция для загрузки FBX-файла в сцену
def load_fbx_scene(filename):
    importer = fbx.FbxImporter.Create(manager, "")
    scene = fbx.FbxScene.Create(manager, filename)
    importer.Initialize(filename, -1, manager.GetIOSettings())
    importer.Import(scene)
    importer.Destroy()
    return scene


# Загружаем анимацию и модель
scene_anim = load_fbx_scene("animations/animated_output3.fbx")
scene_model = load_fbx_scene("models/78466.fbx")


# Функция для поиска скелета в сцене
def find_skeleton(node):
    if node.GetNodeAttribute() and node.GetNodeAttribute().GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton:
        return node
    for i in range(node.GetChildCount()):
        skel = find_skeleton(node.GetChild(i))
        if skel:
            return skel
    return None


# Находим корневую кость (скелет) в анимации
root_skeleton = find_skeleton(scene_anim.GetRootNode())


# Находим мэш в модели
def find_mesh(node):
    if node.GetNodeAttribute() and node.GetNodeAttribute().GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
        return node
    for i in range(node.GetChildCount()):
        mesh = find_mesh(node.GetChild(i))
        if mesh:
            return mesh
    return None


mesh_node = find_mesh(scene_model.GetRootNode())

if not root_skeleton or not mesh_node:
    print("❌ Ошибка: Не найден скелет или мэш!")
else:
    print("✅ Найдены скелет и мэш! Начинаем привязку...")

    # Привязываем мэш к скелету (делаем скелет родителем)
    scene_anim.GetRootNode().AddChild(mesh_node)

    # Создаём скин-объект для мэша
    skin = fbx.FbxSkin.Create(manager, "Skin")

    # Привязываем все кости к скиннингу
    def add_bones_to_skin(skin, node):
        if node.GetNodeAttribute() and node.GetNodeAttribute().GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton:
            cluster = fbx.FbxCluster.Create(manager, node.GetName() + "_Cluster")
            cluster.SetLink(node)
            cluster.SetLinkMode(fbx.FbxCluster.ELinkMode.eTotalOne)  # Полное влияние одной кости
            skin.AddCluster(cluster)

        for i in range(node.GetChildCount()):
            add_bones_to_skin(skin, node.GetChild(i))


    add_bones_to_skin(skin, root_skeleton)

    # Добавляем скин к мешу
    mesh = mesh_node.GetNodeAttribute()
    mesh.AddDeformer(skin)

    # Экспортируем обновлённый FBX
    exporter = fbx.FbxExporter.Create(manager, "")
    exporter.Initialize("output.fbx", -1, manager.GetIOSettings())
    exporter.Export(scene_anim)
    exporter.Destroy()

    print("✅ Экспорт завершён! Файл сохранён как output.fbx")

import fbx


def get_skeleton_nodes(fbx_file_path):
    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)

    importer = fbx.FbxImporter.Create(manager, "")
    scene = fbx.FbxScene.Create(manager, "scene")

    if not importer.Initialize(fbx_file_path, -1, manager.GetIOSettings()):
        raise RuntimeError("Failed to initialize FBX importer")

    if not importer.Import(scene):
        raise RuntimeError("Failed to import FBX scene")

    importer.Destroy()

    skeleton_nodes = {}

    def traverse(node_):
        attr = node_.GetNodeAttribute()
        if attr and (attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton or node_.GetName() == 'root'):
            skeleton_nodes[node_.GetName()] = node_
        for ind in range(node_.GetChildCount()):
            traverse(node_.GetChild(ind))

    root = scene.GetRootNode()
    if root:
        for i in range(root.GetChildCount()):
            traverse(root.GetChild(i))

    return scene, skeleton_nodes


if __name__ == '__main__':
    fbx_path = "models/spider_man_model3.fbx"
    _, nodes = get_skeleton_nodes(fbx_path)

    print(len(nodes))

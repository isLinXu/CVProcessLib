import open3d as o3d

def Visualizes_pointcloud(file_path): #Visualizes scene.pcd
    '''
    可视化.pcd/ply点云文件
    Visualizes file.pcd
    :param file_path:
    :return:
    '''
    PntCld = o3d.io.read_point_cloud(file_path)
    print(PntCld)
    o3d.visualization.draw_geometries([PntCld])


if __name__ == '__main__':
    # 预览显示pcd/ply点云文件
    # pcd_path = 'tpcd/pc.ply'
    pcd_path = 'tpcd/scene.pcd'
    Visualizes_pointcloud(pcd_path)
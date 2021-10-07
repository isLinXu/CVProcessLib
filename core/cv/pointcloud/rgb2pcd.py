import open3d as o3d

def save_pointcloud(img_color_path,img_depth_path, filename='pointcloud'):
    '''
    读取RGB+D图像重建保存为pcd/ply点云
    :param img_color_path:
    :param img_depth_path:
    :param filename:
    :return:
    '''
    #Reading Images 分别读取图像
    img_color = o3d.io.read_image(img_color_path)
    img_depth = o3d.io.read_image(img_depth_path)

    #Creating 创建RGBD
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(img_color, img_depth)

    #create point cloud 创建点云对象
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # Flipping the pointcloud, or it will be upside down 翻转点云
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    #size of cube 设置立方体大小
    size_cube = 0.5
    # pick_points(pcd)
    [gx, gy, gz] = [-0.28, -0.85+size_cube/2, -3.7] #point on the 'ground'

    # create co-ordinate frame 创建坐标系
    # size of coordinate frame 设置坐标系的大小
    crd_size = 0.55
    A = o3d.geometry.TriangleMesh.create_coordinate_frame(size = crd_size, origin = [-gx,-gy,-gz])

    # create file.pcd[world frm + pntcld] 保存点云文件
    o3d.io.write_point_cloud(filename+".pcd", pcd + A.sample_points_uniformly(number_of_points= 500))
    print('save success')

if __name__ == '__main__':
    # 重建保存pcd/ply点云文件
    img_color_path = '/home/hxzh02/文档/pyrgbd/data/tum/color.png'
    img_depth_path = '/home/hxzh02/文档/pyrgbd/data/tum/depth.png'
    # img_color_path = 'timg/color.jpg'
    # img_depth_path = 'timg/depth.png'
    filename = 'room'
    save_pointcloud(img_color_path,img_depth_path,filename)

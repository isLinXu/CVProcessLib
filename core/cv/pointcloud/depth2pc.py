import cv2
import numpy as np
import sys
import os
import time
sys.path.append(os.path.abspath('.'))
# import pyrgbd


def unproject(u, v, d, fx, fy, cx, cy):
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    # for scalar and tensor
    return np.stack([x, y, d], axis=-1)


# https://docs.opencv.org/4.3.0/d9/d0c/group__calib3d.html
def _undistort_pixel_opencv(u, v, fx, fy, cx, cy, k1, k2, p1, p2,
                            k3=0.0, k4=0.0, k5=0.0, k6=0.0):
    # https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/undistort.cpp#L345
    # https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L385
    x0 = (u - cx) / fx
    y0 = (v - cy) / fy
    x = x0
    y = y0
    # Compensate distortion iteratively
    # 5 is from OpenCV code.
    # I don't know theoritical rationale why 5 is enough...
    max_iter = 5
    for j in range(max_iter):
        r2 = x * x + y * y
        icdist = (1 + ((k6 * r2 + k5) * r2 + k4) * r2) / \
                        (1 + ((k3 * r2 + k2) * r2 + k1) * r2)
        deltaX = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        deltaY = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (x0 - deltaX) * icdist
        y = (y0 - deltaY) * icdist

    u_ = x * fx + cx
    v_ = y * fy + cy

    return u_, v_

def undistort_pixel(u, v, fx, fy, cx, cy, distortion_type, distortion_param):
    if distortion_type == "OPENCV" and 4 <= len(distortion_param) <= 8:
        # k1, k2, p1, p2 = distortion_param
        return _undistort_pixel_opencv(u, v, fx, fy,
                                       cx, cy, *tuple(distortion_param))
    raise NotImplementedError(
        distortion_type + " with param " + distortion_param
        + " is not implemented")

def depth2pc(depth, fx, fy, cx, cy, color=None, ignore_zero=True,
             keep_image_coord=False,
             distortion_type=None, distortion_param=[],
             distortion_interp='NN'):
    if depth.ndim != 2:
        raise ValueError()
    with_color = color is not None
    with_distortion = distortion_type is not None
    if ignore_zero:
        valid_mask = depth > 0
    else:
        valid_mask = np.ones(depth.shape, dtype=np.bool)
    invalid_mask = np.logical_not(valid_mask)
    h, w = depth.shape
    u = np.tile(np.arange(w), (h, 1))
    v = np.tile(np.arange(h), (w, 1)).T
    if with_distortion:
        u, v = undistort_pixel(u, v, fx, fy, cx, cy,
                               distortion_type, distortion_param)
    pc = unproject(u, v, depth, fx, fy, cx, cy)
    pc[invalid_mask] = 0

    pc_color = None
    if with_color:
        # interpolation for float uv by undistort_pixel
        if with_distortion and distortion_interp == 'NN':
            v, u = np.rint(v), np.rint(u)
        elif with_distortion:
            raise NotImplementedError('distortion_interp ' +
                                      distortion_interp +
                                      ' is not implemented')

        # 1) Make UV valid mask for color
        v_valid = np.logical_and(0 <= v, v < h)
        u_valid = np.logical_and(0 <= u, u < w)
        uv_valid = np.logical_and(u_valid, v_valid)

        # 2) Set stub value for outside of valid mask
        v[v < 0] = 0
        v[(h - 1) < v] = h - 1
        u[u < 0] = 0
        u[(w - 1) < u] = w - 1
        pc_color = color[v, u]

        # 3) Update valid_mask and invalid_mask
        valid_mask = np.logical_and(valid_mask, uv_valid)
        invalid_mask = np.logical_not(valid_mask)

        pc_color[invalid_mask] = 0

    # return as organized point cloud keeping original image shape
    if keep_image_coord:
        return pc, pc_color

    # return as a set of points
    return pc[valid_mask], pc_color[valid_mask]

def write_pc_ply_txt(path, pc, color=[], normal=[]):
    with open(path, 'w') as f:
        txt = _make_ply_txt(pc, color, normal)
        f.write(txt)

def _make_ply_txt(pc, color, normal):
    header_lines = ["ply", "format ascii 1.0",
                    "element vertex " + str(len(pc)),
                    "property float x", "property float y", "property float z"]
    has_normal = len(pc) == len(normal)
    has_color = len(pc) == len(color)
    if has_normal:
        header_lines += ["property float nx",
                         "property float ny", "property float nz"]
    if has_color:
        header_lines += ["property uchar red", "property uchar green",
                         "property uchar blue", "property uchar alpha"]
    # no face
    header_lines += ["element face 0",
                     "property list uchar int vertex_indices", "end_header"]
    header = "\n".join(header_lines) + "\n"

    data_lines = []
    for i in range(len(pc)):
        line = [pc[i][0], pc[i][1], pc[i][2]]
        if has_normal:
            line += [normal[i][0], normal[i][1], normal[i][2]]
        if has_color:
            line += [int(color[i][0]), int(color[i][1]), int(color[i][2]), 255]
        line_txt = " ".join([str(x) for x in line])
        data_lines.append(line_txt)
    data_txt = "\n".join(data_lines)

    # no face
    ply_txt = header + data_txt

    return ply_txt

def color_depth2ply(img_color_path,img_depth_path,filename ):
    # color
    color = cv2.imread(img_color_path, -1)
    color = color[:, :, [2, 1, 0]]  # BGR to RGB
    # depth
    depth = cv2.imread(img_depth_path, -1)  # 16bit short
    depth = depth.astype(np.float32)
    depth /= 5000.0  # resolve TUM depth scale and convert to meter scale
    # intrinsics of Freiburg 3 RGB
    fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
    start_t = time.time()
    pc, pc_color = depth2pc(depth, fx, fy, cx, cy, color,
                            keep_image_coord=False)
    end_t = time.time()
    print('depth2pc: {:.1f} ms'.format((end_t - start_t) * 1000.0))

    start_t = time.time()
    write_pc_ply_txt('tpcd/' + filename + '.ply', pc, pc_color)
    end_t = time.time()
    print('write_pc_ply_txt: {:.1f} ms'.format((end_t - start_t) * 1000.0))
    print('save success!')



if __name__ == '__main__':
    img_color_path = '/home/hxzh02/文档/pyrgbd/data/tum/room_color.png'
    img_depth_path = '/home/hxzh02/文档/pyrgbd/data/tum/room_depth.png'
    filename = 'rm'
    color_depth2ply(img_color_path, img_depth_path, filename)
from numpy.lib.stride_tricks import as_strided
import numpy as np
import torch
# from visdom import Visdom
from argparse import Namespace
import cv2
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
# import torch
# from rearrange_on_proc.arguments import args
# import json
# from heapq import heappush, heappop
# from rearrange_on_proc.constants import IOU_THRESHOLD, OPENNESS_THRESHOLD, POSITION_DIFF_BARRIER, CATEGORY_LIST, SEGMENTATION_CATEGORY_to_COLOR
#
# import traceback
# from allenact.utils.system import get_logger
# from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
# from rearrange_on_proc.rearrange_challenge.utils import (
#     get_pose_info,
#     iou_box_3d,
# )
# from typing import Dict, Any, Tuple, Optional, Callable, List, Union, Sequence
# import lru
# import ai2thor
# from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
# from collections import OrderedDict, deque
# import re


def calculate_iou_numpy(array1, array2):
    intersection = np.sum(array1 * array2)
    union = np.sum(np.logical_or(array1, array2))
    iou = intersection / union
    return iou


def euclidean_distance(a, b):
    """计算欧几里得距离"""
    return np.linalg.norm(a-b)


def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov."""
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
    camera_matrix = {'xc': xc, 'zc': zc, 'f': f}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix


# class Foo(object):
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#
#     def __str__(self):
#         str_ = ''
#         for v in vars(self).keys():
#             a = getattr(self, v)
#             if True:  # isinstance(v, object):
#                 str__ = str(a)
#                 str__ = str__.replace('\n', '\n  ')
#             else:
#                 str__ = str(a)
#             str_ += '{:s}: {:s}'.format(v, str__)
#             str_ += '\n'
#         return str_
#
#
def get_point_cloud_from_z(Y, camera_matrix):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
      Y is ...xHxW (in our case: 480*480)
      camera_matrix
    Outputs:
      X is positive going right
      Y is positive into the image
      Z is positive up in the image
      XYZ is ...xHxWx3
    """
    x, z = np.meshgrid(np.arange(Y.shape[-1]),
                       np.arange(Y.shape[-2] - 1, -1, -1))
    for i in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (x - camera_matrix.xc) * Y / camera_matrix.f
    Z = (z - camera_matrix.zc) * Y / camera_matrix.f
    XYZ = np.concatenate((X[..., np.newaxis], Y[..., np.newaxis],
                          Z[..., np.newaxis]), axis=X.ndim)
    return XYZ
#
#
def transform_point_cloud_to_egocentric(XYZ, sensor_height, camera_elevation_degree):
    """Transforms the point cloud into geocentric coordinate frame.get_r_matrix
    Input:
      XYZ                     : ...x3
      sensor_height           : height of the sensor
      camera_elevation_degree : camera elevation to rectify.
    Output:
      XYZ : ...x3
    """
    R = get_r_matrix([1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ
#
#
def normalize(v):
    return v / np.linalg.norm(v)
#
#
ANGLE_EPS = 0.001
#
#
def get_r_matrix(ax_, angle):
    '''利用角轴旋转公式计算旋转矩阵: 罗德里格斯旋转公式 (right hand rule)
       ax_:旋转轴
       angle:旋转角
    '''
    ax = normalize(ax_)
    if np.abs(angle) > ANGLE_EPS:
        S_hat = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]],
                         [-ax[1], ax[0], 0.0]], dtype=np.float32)
        R = np.eye(3) + np.sin(angle) * S_hat + \
            (1 - np.cos(angle)) * (np.linalg.matrix_power(S_hat, 2))
    else:
        R = np.eye(3)
    return R
#
#
def bin_points(XYZ_ms, map_size, z_bins, xy_resolution):
    """Bins points into xy-z bins
    XYZ_ms is ... x H x W x3
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    这是一个把点数据分组并统计数量的函数。该函数将输入的点数据 XYZ_ms 按照 x、y 和 z 轴坐标划分到 xy-z 直方图中
    """
    sh = XYZ_ms.shape
    XYZ_ms = XYZ_ms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    n_z_bins = len(z_bins) + 1
    counts = []
    isvalids = []
    inds = []
    for XYZ_m in XYZ_ms:
        isnotnan = np.logical_not(np.isnan(XYZ_m[:, :, 0]))
        X_bin = np.round(XYZ_m[:, :, 0] / xy_resolution).astype(np.int32)
        Y_bin = np.round(XYZ_m[:, :, 1] / xy_resolution).astype(np.int32)
        Z_bin = np.digitize(XYZ_m[:, :, 2], bins=z_bins).astype(
            np.int32)  # 按照z_bins区间划分高度

        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0, Y_bin < map_size,
                            Z_bin >= 0, Z_bin < n_z_bins, isnotnan])  # H*W*7

        # 坐标是否有效（即坐标是否在 xy-z 平面内，并且点是否有效，未包含 NaN 值）480*480
        isvalid = np.all(isvalid, axis=0)  # H*W
        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
        ind[np.logical_not(isvalid)] = 0  # H*W个在地图上的索引
        count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32),
                            minlength=map_size * map_size * n_z_bins)
        # 对有效点的index（对应3D地图）进行统计
        count = np.reshape(count, [map_size, map_size, n_z_bins])
        counts.append(count)
        isvalids.append(isvalid)
        # ？lwj:  不是很懂为什么这里要把所有的index都减小H*W
        ind = ind - (ind.shape[0] * ind.shape[1])
        ind[np.logical_not(isvalid)] = -1
        inds.append(ind)
        counts = np.array(counts).reshape(
        list(sh[:-3]) + [map_size, map_size, n_z_bins])
    isvalids = np.array(isvalids).reshape(list(sh[:-3]) + [sh[-3], sh[-2], 1])
    inds = np.array(inds).reshape(list(sh[:-3]) + [sh[-3], sh[-2], 1])
    return counts, isvalids, inds
#
# # 计算并返回单个 4x4 矩阵的逆矩阵
#
#
# def safe_inverse_single(a):
#     r, t = split_rt_single(a)
#     t = t.view(3, 1)
#     r_transpose = r.t()
#     inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
#     bottom_row = a[3:4, :]  # this is [0, 0, 0, 1]
#     # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4)
#     inv = torch.cat([inv, bottom_row], 0)
#     return inv
#
#
# def split_rt_single(rt):
#     r = rt[:3, :3]
#     t = rt[:3, 3].view(3)
#     return r, t
#
#
# def standardization(data):  # 标准化变成-1到1
#     mu = np.mean(data, axis=0)
#     sigma = np.std(data, axis=0)
#     return (data - mu) / sigma
#
#
# def normalization(data):  # 归一化变成0-1
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range
#
# # 用于中途测试可视化
#
#
def visdomImage(imgs, vis, tag, mapper=None, info=None, win_name=None, max_depth=None, prev_action=None,
                current_step=None, title: str = None, point_goal=None, start_point=None, path=None, map_for_distance = None):
    if tag == 'subplot':
        plt.figure(1, figsize=(10, 4))
        n = len(imgs)
        plt.clf()
        for i in range(n):
            img = imgs[i]
            ax = plt.subplot(1, n, i + 1)
            if info[i] == 'map_bool':  # map(traversible)
                plt.imshow(img * 0.8, origin='lower', cmap='Greys', vmin=0, vmax=1)
                norm = plt.Normalize(vmin = 0, vmax = 6)
                # print('----------------------', np.nanmax(map_for_distance))
                plt.imshow(map_for_distance, alpha = 0.6, origin='lower', interpolation='nearest', cmap=plt.get_cmap('coolwarm'), norm = norm)
                ax.set_title('-traversible')
            elif info[i] == 'map_localmap':  # map(local map)
                img = img / 100 * 255
                img = img.astype(np.uint8)
                img = cv2.applyColorMap(img, 2)
                img = Image.fromarray(img)
                plt.imshow(img, origin='lower')
                ax.set_title('local map')
            elif info[i] == 'depth':  # depth
                d = img * 1.
                d[d > max_depth] = 0
                # d = normalization(d) * 255
                d = d / max_depth * 255
                d = d.astype(np.uint8)
                d_color = cv2.applyColorMap(d, 2)
                d_color = Image.fromarray(d_color)
                plt.imshow(d_color)
                # ax.set_title('depth')
            elif info[i] == 'rgb':  # rgb
                plt.imshow(img)
                ax.set_title(f"steps:{current_step} last action:{prev_action}")
            elif info[i] == 'rgb_seg':
                plt.imshow(img)
            elif info[i] == 'map_seg_topdown':
                plt.imshow(img, origin='lower')
                ax.set_title(f"seg_map")
            elif info[i] == 'map_high_obs_topdown':
                img = img / 17 * 255   # len(z_bins) = 17
                img = img.astype(np.uint8)
                img_color = cv2.applyColorMap(img, 2)
                img_color = Image.fromarray(img_color)
                plt.imshow(img_color, origin='lower')
            elif info[i] == 'map_seg_hierarchical':
                category = CATEGORY_LIST[35-1] #Box
                category_color = SEGMENTATION_CATEGORY_to_COLOR[category]
                category_color = RGB_to_Hex(category_color)
                colormap = get_colormap(name=category, color_list=['#FFFFFF', category_color])
                plt.imshow(img, cmap=colormap, origin='lower')
                ax.set_title('SideTable seg')
            elif info[i] == 'seg_mask':
                plt.imshow(img)
                ax.set_title('SideTable mask')

    elif tag == '01':
        fig = plt.figure(1, dpi=args.dpi)
        plt.title(title)
        if(point_goal != None):
            plt.plot(point_goal[1], point_goal[0], color='blue',
                     marker='o', linewidth=10, markersize=12)
        plt.imshow(imgs, cmap='Greys')
        # vis.images(img, win=win_name)
    elif tag == 'dij':
        plt.figure()
        plt.imshow(imgs, cmap='binary')
        plt.grid(linewidth=1)
        plt.xticks(np.arange(0, 125, 1))
        plt.yticks(np.arange(0, 125, 1))
        if path != None:
            for i in range(len(path)):
                x, y = path[i]
                plt.plot(y, x, 'ro')
        # print("!!!!!!!!!!!!!!!!!!", path)
        plt.plot(start_point[1], start_point[0], 'go')
        plt.plot(point_goal[1], point_goal[0], 'bo')
        plt.savefig('/home/yyl/lyy/rearrange_on_ProcTHOR/' +
                    str(start_point) + 'to' + str(point_goal) + '.png')
        # plt.show()
    vis.matplot(plt, win=win_name)
    print('==>visdom draw image')
#
#
# def get_colormap(name, color_list, N=256):
#     new_colormap = LinearSegmentedColormap.from_list(
#         name, colors=color_list, N=N)
#     return new_colormap
#
#
# def RGB_to_Hex(RGB):
#     color = '#'
#     for i in RGB:
#         num = int(i * 255.)
#         # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
#         color += str(hex(num))[-2:].replace('x', '0').upper()
#     return color
#
#
# class MyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(MyEncoder, self).default(obj)
#
#
# def pool2d(A: np.ndarray, kernel_size: int, stride: int, pool_mode: str = 'avg') -> np.ndarray:
#     '''
#     2D Pooling
#     Parameters:
#         A: input 2D array
#         kernel_size: int, the size of the window over which we take pool
#         stride: int, the stride of the window
#         pool_mode: string, 'max' or 'avg'
#     '''
#     # Window view of A
#     output_shape = ((A.shape[0] - kernel_size) // stride + 1,
#                     (A.shape[1] - kernel_size) // stride + 1)
#     shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
#     strides_w = (stride * A.strides[0], stride *
#                  A.strides[1], A.strides[0], A.strides[1])
#     A_w = as_strided(A, shape_w, strides_w)
#     # Return the result of pooling
#     if pool_mode == 'max':
#         return A_w.max(axis=(2, 3))
#     elif pool_mode == 'avg':
#         return A_w.mean(axis=(2, 3))
#
#
# def dijkstra(grid: np.ndarray, start, end):
#     '''
#     在网格地图上用dijkstra算法计算最短路径
#     input:
#     grid:地图，True表示可以通行，False表示不能通过
#     start：起始点
#     end：终止点
#
#     output：
#     path：路径上所有点组成的列表
#     '''
#     rows, cols = grid.shape
#     dist = np.zeros((rows, cols)) + np.inf
#     dist[start] = 0
#     heap = [(0, start)]
#     prior_node = np.zeros((rows, cols), dtype=tuple)
#     prior_node[start] = start
#     path = []
#     while heap:
#         d, curr = heappop(heap)
#         if curr == end:
#             break
#         for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#             x, y = curr[0] + dx, curr[1] + dy
#             if 0 <= x < rows and 0 <= y < cols and grid[x, y] and d + 1 < dist[x, y]:
#                 dist[x, y] = d + 1
#                 heappush(heap, (d + 1, (x, y)))
#                 prior_node[x, y] = curr
#     # print(prior_node)
#     curr = end
#     if curr == start:
#         return None
#     while curr != start:
#         # print("curr",curr)
#         if curr == 0:
#             return None
#         x, y = curr
#         path.append(curr)
#         curr = prior_node[x, y]
#     # 有一些小问题，当start和end相同时，path里面只有一个点，就是start
#     path.append(curr)
#     path = list(reversed(path))
#     return path
#
#
# def round_to_factor(num: float, base: int) -> int:
#     """Rounds floating point number to the nearest integer multiple of the
#     given base. E.g., for floating number 90.1 and integer base 45, the result
#     is 90.
#
#     # Attributes
#
#     num : floating point number to be rounded.
#     base: integer base
#     """
#     return round(num / base) * base
#
#
# def compare_poses(goal_pose, cur_pose):
#     assert goal_pose["type"] == cur_pose["type"]
#     assert not goal_pose["broken"]
#
#     if cur_pose["broken"]:
#         return {
#             "broken": True,
#             "iou": None,
#             "openness_diff": None,
#             "position_dist": None,
#             "rotation_dist": None,
#         }
#
#     if goal_pose["bounding_box"] is None and cur_pose["bounding_box"] is None:
#         iou = None
#         position_dist = None
#         rotation_dist = None
#     else:
#         position_dist = IThorEnvironment.position_dist(
#             goal_pose["position"], cur_pose["position"]
#         )
#         rotation_dist = IThorEnvironment.angle_between_rotations(
#             goal_pose["rotation"], cur_pose["rotation"]
#         )
#         if position_dist < 1e-2 and rotation_dist < 10.0:
#             iou = 1.0
#         else:
#             try:
#                 iou = iou_box_3d(
#                     goal_pose["bounding_box"], cur_pose["bounding_box"]
#                 )
#             except Exception as _:
#                 get_logger().warning(
#                     "Could not compute IOU, will assume it was 0. Error during IOU computation:"
#                     f"\n{traceback.format_exc()}"
#                 )
#                 iou = 0
#
#     if goal_pose["openness"] is None and cur_pose["openness"] is None:
#         openness_diff = None
#     else:
#         openness_diff = abs(goal_pose["openness"] - cur_pose["openness"])
#
#     return {
#         "broken": False,
#         "iou": iou,
#         "openness_diff": openness_diff,
#         "position_dist": position_dist,
#         "rotation_dist": rotation_dist,
#     }
#
#
# def pose_difference_energy(goal_poses, cur_poses):
#     '''
#         Input: {id: pose_info{}; id: pose_info{}}
#         Output: {id: energy; id: energy}
#     '''
#     goal_pose_objs_id = set(goal_poses.keys())
#     cur_pose_objs_id = set(cur_poses.keys())
#     assert goal_pose_objs_id == cur_pose_objs_id
#
#     min_iou: float = IOU_THRESHOLD
#     open_tol: float = OPENNESS_THRESHOLD
#     pos_barrier: float = POSITION_DIFF_BARRIER
#
#     obj_id_to_energy = {}
#     for obj_id in goal_pose_objs_id:
#         goal_pose = get_pose_info((goal_poses[obj_id]))
#         cur_pose = get_pose_info((cur_poses[obj_id]))
#         assert not goal_pose["broken"]
#
#         pose_diff = compare_poses(goal_pose=goal_pose, cur_pose=cur_pose)
#         if pose_diff["broken"]:
#             energy = 1.0
#
#         elif pose_diff["openness_diff"] is None or goal_pose["pickupable"]:
#             gbb = np.array(goal_pose["bounding_box"])
#             cbb = np.array(cur_pose["bounding_box"])
#
#             iou = pose_diff["iou"]
#             # print('iou:', obj_id, iou, type(iou), len(iou) if type(iou) == tuple else None)
#             iou_energy = max(1 - iou / min_iou, 0)
#
#             if iou > 0:
#                 position_dist_energy = 0.0
#             else:
#                 min_pairwise_dist_between_corners = np.sqrt(
#                     (
#                         (
#                             np.tile(gbb, (1, 8)).reshape(-1, 3)
#                             - np.tile(cbb, (8, 1)).reshape(-1, 3)
#                         )
#                         ** 2
#                     ).sum(1)
#                 ).min()
#                 position_dist_energy = min(
#                     min_pairwise_dist_between_corners / pos_barrier, 1.0
#                 )
#
#             energy = 0.5 * iou_energy + 0.5 * position_dist_energy
#
#         else:
#             energy = 1.0 * (pose_diff["openness_diff"] > open_tol)
#             # print(type(pose_diff['openness_diff']), type(open_tol))
#
#         obj_id_to_energy[obj_id] = energy
#
#     obj_id_to_energy = {k: obj_id_to_energy[k]
#                         for k in sorted(obj_id_to_energy.keys())}
#
#     return obj_id_to_energy
#
#
# # def _obj_list_to_obj_name_to_pose_dict(objects: List[Dict[str, Any]]) -> OrderedDict:
# #     """Helper function to transform a list of object data dicts into a
# #     dictionary."""
# #     objects = [
# #         o
# #         for o in objects
# #         if o["openable"] or o.get("objectOrientedBoundingBox") is not None
# #     ]
# #     d = OrderedDict(
# #         (o["name"], o) for o in sorted(objects, key=lambda x: x["name"])
# #     )
# #     # lwj test procTHOR
# #     # return d
# #     assert len(d) == len(objects)
# #     return d
#
# def hand_in_initial_position(
#     controller: ai2thor.controller.Controller, ignore_rotation: bool = False
# ):
#     metadata = controller.last_event.metadata
#     return IThorEnvironment.position_dist(
#         metadata["heldObjectPose"]["localPosition"], {
#             "x": 0, "y": -0.16, "z": 0.38},
#     ) < 1e-4 and (
#         ignore_rotation
#         or IThorEnvironment.angle_between_rotations(
#             metadata["heldObjectPose"]["localRotation"],
#             {"x": -metadata["agent"]["cameraHorizon"], "y": 0, "z": 0},
#         )
#         < 1e-2
#     )
#
#
# class ObjectInteractablePostionsCache:
#     def __init__(self, max_size: int = 20000, ndigits=2):
#         self._key_to_positions = lru.LRU(size=max_size)
#
#         self.ndigits = ndigits
#         self.max_size = max_size
#
#     def _get_key(self, scene_name: str, obj: Dict[str, Any]):
#         p = obj["position"]
#         return (
#             scene_name,
#             obj["type"] if "type" in obj else obj["objectType"],
#             round(p["x"], self.ndigits),
#             round(p["y"], self.ndigits),
#             round(p["z"], self.ndigits),
#         )
#
#     def get(
#         self,
#         scene_name: str,
#         obj: Dict[str, Any],
#         controller: ai2thor.controller.Controller,
#         reachable_positions: Optional[Sequence[Dict[str, float]]] = None,
#         force_cache_refresh: bool = False,
#     ) -> List[Dict[str, Union[float, int, bool]]]:
#
#         obj_key = self._get_key(scene_name=scene_name, obj=obj)
#
#         if force_cache_refresh or obj_key not in self._key_to_positions:
#             metadata = controller.last_event.metadata
#
#             obj_in_scene = next(
#                 (o for o in metadata["objects"]
#                  if o["name"] == obj["name"]), None,
#             )
#             if obj_in_scene is None:
#                 raise RuntimeError(
#                     f"Object with name {obj['name']} must be in the scene when filling a cache miss"
#                 )
#
#             desired_pos = obj["position"]
#             desired_rot = obj["rotation"]
#
#             cur_pos = obj_in_scene["position"]
#             cur_rot = obj_in_scene["rotation"]
#
#             should_teleport = (
#                 IThorEnvironment.position_dist(desired_pos, cur_pos) >= 1e-3
#                 or IThorEnvironment.rotation_dist(desired_rot, cur_rot) >= 1
#             )
#
#             object_held = obj_in_scene["isPickedUp"]
#             physics_was_unpaused = controller.last_event.metadata.get(
#                 "physicsAutoSimulation", True
#             )
#             if should_teleport:
#                 if object_held:
#                     if not hand_in_initial_position(
#                         controller=controller, ignore_rotation=True
#                     ):
#                         raise NotImplementedError
#
#                     if physics_was_unpaused:
#                         controller.step("PausePhysicsAutoSim")
#                         assert controller.last_event.metadata["lastActionSuccess"]
#
#                 event = controller.step(
#                     "TeleportObject",
#                     objectId=obj_in_scene["objectId"],
#                     rotation=desired_rot,
#                     **desired_pos,
#                     forceAction=True,
#                     allowTeleportOutOfHand=True,
#                     forceKinematic=True,
#                 )
#                 assert event.metadata["lastActionSuccess"]
#
#             metadata = controller.step(
#                 action="GetInteractablePoses",
#                 objectId=obj["objectId"],
#                 positions=reachable_positions,
#                 horizons=np.arange(-30, 60 + 1, args.HORIZON_DT),
#             ).metadata
#             assert metadata["lastActionSuccess"]
#             self._key_to_positions[obj_key] = metadata["actionReturn"]
#
#             if should_teleport:
#                 if object_held:
#                     if hand_in_initial_position(
#                         controller=controller, ignore_rotation=True
#                     ):
#                         controller.step(
#                             "PickupObject",
#                             objectId=obj_in_scene["objectId"],
#                             forceAction=True,
#                         )
#                         assert controller.last_event.metadata["lastActionSuccess"]
#
#                         if physics_was_unpaused:
#                             controller.step("UnpausePhysicsAutoSim")
#                             assert controller.last_event.metadata["lastActionSuccess"]
#                     else:
#                         raise NotImplementedError
#                 else:
#                     event = controller.step(
#                         "TeleportObject",
#                         objectId=obj_in_scene["objectId"],
#                         rotation=cur_rot,
#                         **cur_pos,
#                         forceAction=True,
#                     )
#                     assert event.metadata["lastActionSuccess"]
#
#         return self._key_to_positions[obj_key]
#
#
# def are_poses_equal(goal_pose: Dict[str, Any], cur_pose: Dict[str, Any], treat_broken_as_unequal: bool) -> bool:
#     if cur_pose["isBroken"]:
#         if treat_broken_as_unequal:
#             return False
#     position_dist = IThorEnvironment.position_dist(
#         goal_pose["position"], cur_pose["position"])
#     rotation_dist = IThorEnvironment.angle_between_rotations(
#         goal_pose["rotation"], cur_pose["rotation"])
#
#     if position_dist < 1e-2 and rotation_dist < 10.0:
#         return True
#     return False
#
#
# def bfs(grid, start, end):
#     m, n = grid.shape
#     visited = set()
#     queue = deque([(start, 0)])
#     visited.add(start)
#     while queue:
#         (x, y), dist = queue.popleft()
#         if (x, y) == end:
#             return dist
#         for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#             nx, ny = x + dx, y + dy
#             # print("nx: ",nx,"ny: ",ny)
#             if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited and grid[nx][ny] == 1:
#                 visited.add((nx, ny))
#                 queue.append(((nx, ny), dist + 1))
#     return -1  # 如果没有找到目标点，返回 -1
#
#
# def floyd(grid, points):
#     n = len(points)
#     d = np.full((n, n), 10000000, dtype=int)
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 d[i][j] = 0
#             else:
#                 start, end = points[i], points[j]
#                 dist = bfs(grid, start, end)
#                 if dist != -1:
#                     d[i][j] = dist
#     for k in range(n):
#         for i in range(n):
#             for j in range(n):
#                 d[i][j] = min(d[i][j], d[i][k] + d[k][j])
#     return d
#
#
# def format_class_name(self, name):
#     if name == "TVStand":
#         formatted = "television stand"
#     elif name == "CounterTop":
#         formatted = "countertop"
#     else:
#         formatted = re.sub(r"(?<=\w)([A-Z])", r" \1", name).lower()
#     return formatted
#
#
# def get_max_height_on_line(array, point1, point2):
#     # 计算两点之间的横坐标和纵坐标的差值
#     dx, dy = point2[0] - point1[0], point2[1] - point1[1]
#     # 根据两点之间的距离确定采样点的数量
#     num_samples = max(abs(dx), abs(dy)) + 1
#     # 在横坐标上生成等间隔的数据点
#     x = np.linspace(point1[0], point2[0], num=num_samples, endpoint=True)
#     # 在纵坐标上生成等间隔的数据点
#     y = np.linspace(point1[1], point2[1], num=num_samples, endpoint=True)
#     # 沿着第二个维度堆叠生成的数据点
#     coords = np.vstack((x, y)).astype(int)
#     # 获取点的坐标和高度值
#     heights = array[coords[0], coords[1]]
#     max_height = np.max(heights)
#     return max_height
#
#
# def main():
#     # 创建一个5x5的随机高度数组
#     array = np.random.randint(0, 10, size=(5, 5))
#     print(array)
#     # [[3 8 5 9 9]
#     #  [7 2 8 8 7]
#     #  [5 5 5 5 8]
#     #  [9 1 9 7 6]
#     #  [6 7 6 5 6]]
#
#     # 找到点(0,0)和点(4,4)之间的最高点
#     highest_point = get_reachable_points_on_line(array, (0, 0), (4, 4))
#     print(highest_point)
#     # 输出: 9


if __name__ == '__main__':
    main()

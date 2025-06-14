import logging
import numpy as np
import skimage
import skimage.morphology
import pdb 
# from rearrange_on_proc.arguments import args
import ivn_aff.utils.utils as utils
# from rearrange_on_proc.constants import CATEGORY_to_ID, INSTANCE_FEATURE_SIMILARITY_THRESHOLD, INSTANCE_IOU_THRESHOLD
# from torch_scatter import scatter_max
import itertools
import torch
import matplotlib.pyplot as plt
import tkinter
import matplotlib
import matplotlib.gridspec as gridspec
from collections import Counter
# matplotlib.use('TkAgg')

fov = 90.0
map_resolution = 0.05
STEP_SIZE = 0.25


class Mapper():
    def __init__(self, C, origin, map_size, resolution, max_depth=164, z_bins=[0.05, 3], num_categories=50, loc_on_map_selem=skimage.morphology.disk(2)):
        # Internal coordinate frame is X, Y into the scene, Z up.
        self.sc = 1
        self.C = C
        self.resolution = resolution # 清晰度
        self.max_depth = max_depth

        self.z_bins = z_bins
        map_sz = int(np.ceil((map_size*100)//(resolution*100)))
        self.map_sz = map_sz
        # print("MAP SIZE (pixel):", map_sz)

        # 4个地图
        self.map = np.zeros((map_sz, map_sz, len(self.z_bins)+1), dtype=np.float32)
        self.map_for_view_distance = np.ones((map_sz, map_sz), dtype=np.float32) *  float('inf')

        self.local_map = np.zeros((map_sz, map_sz, len(self.z_bins)+1), dtype=np.float32)
        self.semantic_map = np.zeros((map_sz, map_sz, len(self.z_bins)+1, num_categories), dtype=np.float32)
        self.loc_on_map = np.zeros((map_sz, map_sz), dtype=np.float32)
        self.added_obstacles = np.ones((map_sz, map_sz), dtype=bool)

        self.origin_xz = np.array([origin['x'], origin['z']])
        step_pix = int(STEP_SIZE / map_resolution)
        self.origin_map = np.array([(self.map.shape[0]-1-step_pix)/2, (self.map.shape[0]-1-step_pix)/2], np.float32)
        #self.origin_map = self._optimize_set_map_origin(self.origin_xz, self.resolution)
        # self.objects = {}
        self.loc_on_map_selem = loc_on_map_selem

        self.num_boxes = 0

        self.step = 0

        self.bounds = None  # bounds

        #局部点云图
        self.local_point_cloud = None

    # def _optimize_set_map_origin(self, origin_xz, resolution):
    #     return (origin_xz + 15) / resolution

    def transform_egocentric_worldCoordinate(self, XYZ):
        R = utils.get_r_matrix([0., 0., 1.], angle=self.current_rotation)
        XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
        XYZ[:, :, 0] = XYZ[:, :, 0] + self.current_position[0] - self.origin_xz[0] + self.origin_map[0] * self.resolution
        XYZ[:, :, 1] = XYZ[:, :, 1] + self.current_position[1] - self.origin_xz[1] + self.origin_map[1] * self.resolution
        return XYZ

    def update_position_on_map(self, position, rotation):
        self.current_position = np.array(
            [position['x'], position['z']], np.float32)
        self.current_rotation = -np.deg2rad(rotation)
        x, y = self.get_position_on_map()
        self.loc_on_map[int(y), int(x)] = 1
        # TODO(saurabhg): Mark location on map

    def get_position_on_map(self):
        # （地图参考系）现实世界坐标(x, z)(单位:m) --> 地图Map位置（x, y)（单位:pixel）
        map_position = self.current_position - self.origin_xz + self.origin_map * self.resolution  # 单位是 现实世界(m)
        map_position = map_position / self.resolution # 单位是 地图(pixel)
        map_position = [max(map_position[0], 0), max(map_position[1], 0)]
        map_position = [min(map_position[0], self.map_sz-1), min(map_position[1], self.map_sz-1)]
        return map_position
    
    def get_target_on_map(self, target):
        map_position = target - self.origin_xz + self.origin_map * self.resolution  # 单位是 现实世界(m)
        map_position = map_position / self.resolution # 单位是 地图(pixel)
        map_position = [max(map_position[0], 0), max(map_position[1], 0)]
        map_position = [min(map_position[0], self.map_sz-1), min(map_position[1], self.map_sz-1)]
        return map_position

    def add_observation(self, position, rotation, elevation, depth, seg, add_obs=True, add_seg=False, inter_mode=False):

        d = depth * 1.
        d[d > self.max_depth] = 0
        d[d < 0.02] = np.NaN
        d = d / self.sc
        self.update_position_on_map(position, rotation)
        if not add_obs:
            return
        XYZ1 = utils.get_point_cloud_from_z(d, self.C)  # 点云坐标系 （图像坐标系）
        XYZ2 = utils.transform_point_cloud_to_egocentric(XYZ1 * 1, position['y'], elevation)  # 机器人自身坐标系+高度
        XYZ3 = self.transform_egocentric_worldCoordinate(XYZ2)  # 实际模拟器坐标系，以m为单位 [H, W, 3]

        self.local_point_cloud = XYZ3

        counts, is_valids, inds = utils.bin_points(XYZ3, self.map.shape[0], self.z_bins, self.resolution)
        # Shape: counts [map_size, map_size, n_z_bins], is_valids [H, W], inds [H, W]
        self.local_map = counts

        # Shape:
        # seg [H, W, num_category + 1], XYZ3 [H, W, 3]
        # local_semantic [mp_size, mp_size, n_z_bins, num_category + 1]
        if add_seg:
            local_semantic = self.transfrom_feature_to_map(feature=seg, XYZ=XYZ3, xy_resolution=self.resolution,
                                                           z_bins=self.z_bins)
            
        # 如果在交互阶段，刷新agent周围2.5m*2.5m范围的点
        vis = np.zeros((400,400))
        if inter_mode:
            for i in range(local_semantic.shape[0]):
                for j in range(local_semantic.shape[1]):
                    dist = np.sqrt(np.sum(np.square(np.array([j,i]) - self.get_position_on_map())))
                    angle = np.rad2deg(np.arctan2(j-self.get_position_on_map()[0], i-self.get_position_on_map()[1]))
                    d_angle = np.abs((angle%360.0) -rotation)
                    d_angle = d_angle if d_angle <= 180.0 else 360.0 - d_angle
                    if dist <= 60 and d_angle < 55.0:
                        if add_seg:
                            self.semantic_map[i,j,:,:] = local_semantic[i,j,:,:].astype('float')*5
                        self.map[i,j,:] = self.local_map[i,j,:].astype('float')*5
        else:
            self.semantic_map += local_semantic
            self.semantic_map[:,:,:,2] = local_semantic[:,:,:,2]
            self.map += counts
        pass
        # self.semantic_map[:,:,:,2] = local_semantic[:,:,:,2]*5

    def transfrom_feature_to_map(self, feature, XYZ, xy_resolution, z_bins):
        '''
        Input: feature [H, W, feature_dims]. For segmentation,  the 'feature_dims' equals 'num_category+1'
               XYZ [H, W, 3], where '3' represents the coordinates X, Y, Z in the map
        Output: local_feature_map [map_size, map_size, n_z_bins, feature_dims]
        '''
        # 可以直接采用类似 utils.bin_points的方法再加上 feature_dims一维 （默认采用四舍五入）
        # 这里先尝试用FILM里面的邻近八个点插值的方法 (这里考虑xy四点插值+z分bins)
        position_yxz = []
        weight_yxz = []

        feature_dims = feature.shape[-1]
        feature = feature.reshape(-1, feature_dims)
        XYZ[:, :, [0, 1]] = XYZ[:, :, [1, 0]]  # change to (Y, X, Z) for index calculation
        YXZ = XYZ
        H, W, yxz_dims = YXZ.shape
        YXZ = YXZ.reshape(-1, yxz_dims)
        isnotnan = np.logical_not(np.isnan(YXZ[:, 0]))

        map_shape = list(self.map.shape)  # [H,W,len(z_bins)]
        map_shape[0], map_shape[1] = map_shape[1], map_shape[0]  # y,x,z
        n_z_bins = len(z_bins) + 1
        local_feature_map = np.zeros((self.map.shape[0], self.map.shape[1], n_z_bins, feature_dims))
        local_feature_map = torch.from_numpy(local_feature_map).view(-1, feature_dims)

        # yx四点插值+权重
        for somedim in range(yxz_dims - 1):
            position_yxSomedim = YXZ[:, somedim] / xy_resolution
            position_nearInt = []
            weight_nearInt = []

            for dx in [0, 1]:
                position_ix = np.floor(position_yxSomedim) + dx
                isvalid = np.array([position_ix >= 0, position_ix < map_shape[somedim], isnotnan])
                # pdb.set_trace()
                isvalid = np.all(isvalid, axis=0)
                isvalid = isvalid.astype(position_yxSomedim.dtype)

                # pdb.set_trace()
                weight_ix = 1 - np.abs(position_yxSomedim - position_ix)
                weight_ix = weight_ix * isvalid
                position_ix = position_ix * isvalid

                # weight_ix[np.logical_not(isvalid)] = 0
                # position_ix[np.logical_not(isvalid)] = 0

                position_nearInt.append(position_ix)
                weight_nearInt.append(weight_ix)
                # pdb.set_trace()

            position_yxz.append(position_nearInt)
            weight_yxz.append(weight_nearInt)

        Z = np.digitize(YXZ[:, 2], bins=z_bins).astype(np.int32)
        position_yxz.append([Z])
        weight_yxz.append(np.ones_like(Z))

        list_dx = [[0, 1], [0, 1], [0]]
        # 进行笛卡尔积, 考虑xy四点插值+z分bins
        for dy_dx_dz in itertools.product(*list_dx):
            weight = np.ones_like(weight_yxz[0][0])
            index = np.zeros_like(weight_yxz[0][0])
            for somedim in range(yxz_dims):
                index = index * map_shape[somedim] + position_yxz[somedim][dy_dx_dz[somedim]]
                weight = weight * weight_yxz[somedim][dy_dx_dz[somedim]]

            valid_index = np.logical_not(np.isnan(index))
            index_filterNaN = index[valid_index, np.newaxis]
            index_filterNaN_shape = index_filterNaN.shape[0]
            index_filterNaN = index_filterNaN.astype(np.int64)

            index_broadcast = np.broadcast_to(index_filterNaN, (index_filterNaN_shape, feature_dims))
            index_broadcast_tensor = torch.tensor(index_broadcast)
            src_tensor = torch.from_numpy(feature[valid_index] * weight[valid_index, np.newaxis])
            local_feature_map.scatter_add_(0, index_broadcast_tensor, src_tensor)
            local_feature_map = torch.round(local_feature_map)

        local_feature_map = local_feature_map.reshape(self.map.shape[0], self.map.shape[1], n_z_bins,
                                                      feature_dims).numpy().astype(np.int8)
        return local_feature_map
            

    def get_rotation_on_map(self):
        map_rotation = self.current_rotation
        return map_rotation

    def add_obstacle_in_front_of_agent(self,act_name,rotation,size_obstacle=10, pad_width=0,held_mode=False):
        '''
        salem: dilation structure normally used to dilate the map for path planning
        '''
        act_rotation = rotation
        if act_name == 'MoveAhead':
            if held_mode:
                return
        elif act_name == 'MoveLeft':
            act_rotation -= args.DT
        elif act_name == 'MoveRight':
            act_rotation += args.DT
        elif act_name == 'MoveBack':
            act_rotation -= 2*args.DT
        act_rotation = -np.deg2rad(act_rotation)

        size_obstacle = self.loc_on_map_selem.shape[0]  # - erosion_size
        loc_on_map_salem_size = int(np.floor(self.loc_on_map_selem.shape[0]/2))
        


        x, y = self.get_position_on_map()
        # print(self.current_rotation)
        if -np.deg2rad(0) == act_rotation:

            ys = [int(y+loc_on_map_salem_size+1),
                  int(y+loc_on_map_salem_size+1+size_obstacle)]
            y_begin = min(ys)
            y_end = max(ys)
            xs = [int(x-np.floor(size_obstacle/2))-pad_width,
                  int(x+np.floor(size_obstacle/2))+pad_width]
            x_begin = min(xs)
            x_end = max(xs)
        elif -np.deg2rad(90) == act_rotation:
            xs = [int(x+loc_on_map_salem_size+1),
                  int(x+loc_on_map_salem_size+1+size_obstacle)]
            x_begin = min(xs)
            x_end = max(xs)
            ys = [int(y-np.floor(size_obstacle/2))-pad_width,
                  int(y+np.floor(size_obstacle/2))+pad_width]
            y_begin = min(ys)
            y_end = max(ys)
        elif -np.deg2rad(180) == act_rotation:
            ys = [int(y-loc_on_map_salem_size-1),
                  int(y-loc_on_map_salem_size-1-size_obstacle)]
            y_begin = min(ys)
            y_end = max(ys)
            xs = [int(x-np.floor(size_obstacle/2))-pad_width,
                  int(x+np.floor(size_obstacle/2))+pad_width]
            x_begin = min(xs)
            x_end = max(xs)
        elif -np.deg2rad(270) == act_rotation:
            xs = [int(x-loc_on_map_salem_size-1),
                  int(x-loc_on_map_salem_size-1-size_obstacle)]
            x_begin = min(xs)
            x_end = max(xs)
            ys = [int(y-np.floor(size_obstacle/2))-pad_width,
                  int(y+np.floor(size_obstacle/2))+pad_width]
            y_begin = min(ys)
            y_end = max(ys)
        else:
            return
            st()
            assert(False)
        self.added_obstacles[y_begin:y_end, x_begin:x_end] = False

    
    #map上点的数量大于10的记为障碍物
    def get_obstacle(self):
        obstacle = np.sum(self.map[:, :, 1:-3], 2) >= 10
        return obstacle

    # 机器人去过的地方
    def get_visited(self):
        return self.loc_on_map.astype(np.bool8)

    # 机器人去过的地方 or map点小于100的地方 (不考虑bound) H*W
    def get_traversible_map(self, selem, point_count, loc_on_map_traversible):

        #map上点的数量大于100的记为障碍物
        obstacle = self.get_obstacle()
        obstacle = skimage.morphology.binary_dilation(obstacle, selem) == True

        traversible = obstacle != True

        # also add in obstacles
        traversible = np.logical_and(self.added_obstacles, traversible)

        #将机器人去过的地方进行膨胀，及其邻域(一步范围内)都设置为可以到达的地方 # lwj: 其实这里有个小漏洞，领域不一定全是可到达的，但貌似不影响
        if loc_on_map_traversible:

            # traversible_locs = skimage.morphology.binary_dilation(self.loc_on_map, self.loc_on_map_selem) == True
            traversible_locs = self.loc_on_map == True
            traversible = np.logical_or(traversible_locs, traversible) # 机器人去过的地方 or map点小于100的地方

        if self.bounds is not None:
            # limit to scene boundaries
            bounds_x = [self.bounds[0], self.bounds[1]]
            bounds_z = [self.bounds[2], self.bounds[3]]
            len_x_map = int((max(bounds_x) - min(bounds_x))/self.resolution)
            len_z_map = int((max(bounds_z) - min(bounds_z))/self.resolution)
            half_x_map = len_x_map//2
            half_z_map = len_z_map//2
            x_range = [int(self.origin_map[0]-half_x_map),
                       int(self.origin_map[0]+half_x_map)]
            z_range = [int(self.origin_map[1]-half_z_map),
                       int(self.origin_map[1]+half_z_map)]

            traversible[:z_range[0], :] = False
            traversible[z_range[1]:, :] = False
            traversible[:, :x_range[0]] = False
            traversible[:, x_range[1]:] = False

        return traversible

    # lwj:
    def get_local_map_sum(self):
        local_map_sum = np.sum(self.local_map[:, :, 1:-1], 2)
        local_map_sum[local_map_sum > 100] = 100
        return local_map_sum

    def get_map_for_view_distance(self):
        map_for_view_distance = self.map_for_view_distance.copy()
        map_for_view_distance[np.isinf(map_for_view_distance)] = np.nan
        # map_for_view_distance = map_for_view_distance / self.max_depth
        return map_for_view_distance

    def get_explored_map(self, selem, point_count):
        traversible = skimage.morphology.binary_dilation(self.loc_on_map, selem) == True   #准确来说这里的traversible更接近explored这个意思
        # traversible = self.get_traversible_map(selem, point_count, loc_on_map_traversible=True)
        explored = np.sum(self.map, 2) >= point_count       #这里的explored更接近seen的意思, map有点的地方(add_observation得来）
        explored = np.logical_or(explored, traversible)
        if self.bounds is not None:
            # limit to scene boundaries
            bounds_x = [self.bounds[0], self.bounds[1]]
            bounds_z = [self.bounds[2], self.bounds[3]]
            len_x_map = int((max(bounds_x) - min(bounds_x))/self.resolution)
            len_z_map = int((max(bounds_z) - min(bounds_z))/self.resolution)
            half_x_map = len_x_map//2
            half_z_map = len_z_map//2
            x_range = [int(self.origin_map[0]-half_x_map),
                       int(self.origin_map[0]+half_x_map)]
            z_range = [int(self.origin_map[1]-half_z_map),
                       int(self.origin_map[1]+half_z_map)]

            explored[:z_range[0], :] = True
            explored[z_range[1]:, :] = True
            explored[:, :x_range[0]] = True
            explored[:, x_range[1]:] = True
        return explored

    def get_semantic_map_with_maxConfidence_category(self):
        # 根据每个格子的[num_category+1]向量中的点数量来当作confidence
        # transfer the 4D semantic map [H, W, len(z)+1, num_category+1]
        # to 3D semantic map with the max confidence category [H, W, len(z)+1]
        shape = self.semantic_map.shape
        semantic_map = self.semantic_map.reshape(-1, shape[-1])
        
        maxConfidence_indices = np.argmax(semantic_map, axis=1)
        maxConfidence_semantic_map = maxConfidence_indices.reshape(shape[:-1]) # [H, W, len(z)+1]
        return maxConfidence_semantic_map

    def get_semantic_map_with_occupy_category(self):
        # 根据每个格子的[num_category+1]向量中的0/1值表示是否存在该index对应的类别物体；
        # 如果一个格子包含了两个类别，则取index较小的那个【类别】
        # transfer the 4D semantic map [H, W, len(z)+1, num_category+1]
        # to 3D semantic map with the max confidence category [H, W, len(z)+1] 
        shape = self.semantic_map.shape
        semantic_map = self.semantic_map.reshape(-1, shape[-1])
        
        occupy_indices = np.ma.masked_equal(semantic_map, 0).argmin(axis = 1)
        occupy_semantic_map = occupy_indices.reshape(shape[:-1])
        return occupy_semantic_map



    def get_topdown_semantic_map(self):
        # transfer the 4D semantic map to Top-Down 2D semantic map
        # Only for further visualization
        '''
        Input: semantic_map [H, W, len(z) + 1, num_category+1], both H, W refer to the map_size(H/W)
        Note: the category in the higher z_bin will be represented
        '''
        # maxConfidence_semantic_map = self.get_semantic_map_with_maxConfidence_category() #[H, W, len(z)+1]
        maxConfidence_semantic_map = self.get_semantic_map_with_occupy_category()
        shape = maxConfidence_semantic_map.shape
        maxConfidence_semantic_map = maxConfidence_semantic_map.reshape(-1, shape[-1]) # [H*W, len(z)+1]

        z_bins_num = maxConfidence_semantic_map.shape[-1] # 0: no category, 1+: object category
        maxConfidence_semantic_map_flip = np.fliplr(maxConfidence_semantic_map)  # to get the highest z_bin to the first index
        z_bin_with_category = maxConfidence_semantic_map_flip > 0 # get the first non-Zero z_bin (i.e. with category)
        first_z_bin_with_category_index = np.argmax(z_bin_with_category, axis=-1) # [H*W, 1]
        first_z_bin_with_category_index = z_bins_num - 1 - first_z_bin_with_category_index
        no_category = ~np.any(maxConfidence_semantic_map, axis=-1)
        first_z_bin_with_category_index[no_category] = 0

        top_down_category_result = maxConfidence_semantic_map[np.arange(maxConfidence_semantic_map.shape[0]), first_z_bin_with_category_index.ravel()]
        top_down_category_result = top_down_category_result.reshape(shape[0], shape[1])
        
        return top_down_category_result

    def get_hierarchical_semantic_map(self, category_id):
        '''
            Input: category
            Output: the semantic_map of that category
            i.e., transfer the 4D semantic map [H, W, len(z)+1, num_category+1] 
                    To 2D category-wise semantic map [H, W] (with counts)
        '''
        # maxConfidence_semantic_map = self.get_semantic_map_with_maxConfidence_category() #[H, W, len(z)+1]
        category_semantic_map = self.semantic_map[:, :, :, category_id]         #[H, W, len(z)+1]
        # category_semantic_map = np.sum(category_semantic_map, axis=-1) #[H, W] 采用求和（点的数量）
        # category_semantic_map = np.any(category_semantic_map, axis=-1).astype(np.int8)  #【H，W] 采用0/1表示是否存在点
        category_semantic_map = np.amax(category_semantic_map, axis=-1)     #[H, W]
        return category_semantic_map

    def get_topdown_highest_obstacle_map(self):
        # transfer the 3D map to Top-Down 2D highest_obstacle map
        '''
        Input: map [H, W, len(z) + 1], both H, W refer to the map_size(H/W)
        Note: the highest z_bin with obstacle will be represented
        '''
        shape = self.map.shape
        reshaped_map = self.map.reshape(-1, shape[-1])  #[H*W, len(z)+1], each value is the points counts
        z_bins_num = reshaped_map.shape[-1]

        map_flip = np.fliplr(reshaped_map)  # to get the highest z_bin to the first index
        map_flip[:, 0] = 0  # all (>2m, including ceil) are not in consideration
        z_bin_with_point = map_flip > 0  # get the first non-Zero z_bin (i.e. with point counts > 0)
        first_z_bin_with_point_index = np.argmax(z_bin_with_point, axis=-1)  #[H*W, 1]
        first_z_bin_with_point_index = z_bins_num - 1 - first_z_bin_with_point_index
        no_point = ~np.any(map_flip, axis=-1)  #没有点的情况一般指还未探索（未观测到）的，否则至少都有Floor的点
        first_z_bin_with_point_index[no_point] = 0

        first_z_bin_with_point_index = first_z_bin_with_point_index.reshape(shape[0], shape[1])
        return first_z_bin_with_point_index


    def move_map_to_center(self):
        pass

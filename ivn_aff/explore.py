import prior
import numpy as np
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import rearrange_on_proc.utils.utils as utils
import skimage
import torch
import logging
import pdb

from ai2thor.controller import Controller
from PIL import Image
from argparse import Namespace
from arguments import args
from mapper import Mapper
from planner import FMMPlanner
from segmentation import SegmentationHelper

from queue import LifoQueue
from scipy.spatial import distance
from visualization import Animation, visualize_segmentationRGB, visualize_topdownSemanticMap

from rearrange_on_proc.utils.utils import visdomImage, pool2d
from rearrange_on_proc.utils.geom import eul2rotm_py
from rearrange_on_proc.utils.utils import round_to_factor, pose_difference_energy, get_max_height_on_line
from constants import STOP, ROTATE_LEFT, ROTATE_RIGHT, MOVE_AHEAD, MOVE_BACK, MOVE_LEFT, MOVE_RIGHT, DONE, LOOK_DOWN, PICKUP, OPEN, CLOSE, PUT, DROP, LOOK_UP
from constants import POINT_COUNT
from object_tracker import ObjectTrack
from numpy import ma
import skfmm
from torch import nn
import torch.nn.functional as functional


class Explore():
    def __init__(self, controller: Controller, vis: Animation, visdom, curr_stage, map_size, step_max, process_id, solution_config, walkthrough_objs_id_to_pose=None) -> None:
        self.process_id = process_id
        self.controller = controller
        self.curr_stage = curr_stage  # 'walkthrough' or 'unshuffle'
        self.walkthrough_objs_id_to_pose = walkthrough_objs_id_to_pose
        self.step_from_stage_start = 0  # 是可以超过self.step_max的，例如：explore_env()之后的继续整理步数

        if self.curr_stage == 'unshuffle':
            assert self.walkthrough_objs_id_to_pose is not None
            curr_energies_dict = self.get_curr_pose_difference_energy(self.curr_stage)
            curr_energies = np.array(list(curr_energies_dict.values()))
            self.last_energy = curr_energies.sum()
            print(f'unshuffle start_energy: {self.last_energy}')
            # changed_objs_id = [k for k, v in curr_energies_dict.items() if v > 0]
            # changed_objs_parent_goal = [self.walkthrough_objs_id_to_pose[id]['parentReceptacles'] for id in changed_objs_id]
            # changed_objs_parent_curr = [unshuffle_objs_id_to_pose[id]['parentReceptacles'] for id in changed_objs_id]
            # changed = [f"{changed_objs_id[i]}:from {changed_objs_parent_goal[i]} to {changed_objs_parent_curr[i]}" for i in range(len(changed_objs_id))]
            # print(f'unshuffle start_energy: {self.last_energy}. (goal & curr) changed objs: {changed}')
            # print([(k, self.walkthrough_objs_id_to_pose[k], unshuffle_objs_id_to_pose[k]) for k,v in curr_energies_dict.items() if v > 0])

            # self.last_unshuffle_objs_id_to_pose = unshuffle_objs_id_to_pose
        # 每个阶段用的策略
        self.solution_config = solution_config

        # 初始化基础参数
        self.step_max = step_max  # 表示每次执行explore_env()的步数上限
        print('Max Steps: ', self.step_max)

        self.W = args.W
        self.H = args.H
        self.STEP_SIZE = args.STEP_SIZE
        self.HORIZON_DT = args.HORIZON_DT
        self.DT = args.DT
        self.pix_T_camX = None
        # actions = ['MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight',
        #            'RotateRight', 'RotateLeft', 'LookUp', 'LookDown', 'Done']
        # self.act_id_to_name = {i: actions[i] for i in range(len(actions))}
        # 按原TIDEE顺序
        self.act_id_to_name = {
            # STOP: 'RotateLook',
            STOP: 'MoveAhead',
            ROTATE_LEFT: 'RotateLeft',
            ROTATE_RIGHT: 'RotateRight',
            MOVE_AHEAD: 'MoveAhead',
            MOVE_BACK: 'MoveBack',
            MOVE_LEFT: 'MoveLeft',
            MOVE_RIGHT: 'MoveRight',
            DONE: 'Pass',  # lyy注意一下
            LOOK_DOWN: 'LookDown',
            LOOK_UP: 'LookUp',
            PICKUP: 'PickupObject',
            OPEN: 'OpenObject',
            CLOSE: 'CloseObject',
            PUT: 'PutObject',
            DROP: 'DropHandObject',
        }

        # 初始化相机参数
        ar = [args.H, args.W]
        vfov = args.fov * np.pi / 180
        focal = ar[1] / (2 * math.tan(vfov / 2))
        fov = abs(2 * math.atan(ar[0] / (2 * focal)) * 180 / np.pi)
        fov, h, w = fov, ar[1], ar[0]
        C = utils.get_camera_matrix(w, h, fov=fov)

        # 初始化机器人（在地图坐标系）位置和角度
        self.position = {'x': 0,
                         'y': 1.5759992599487305,  # fixed when standing up
                         'z': 0}
        self.head_tilt = round_to_factor(int(
            self.controller.last_event.metadata['agent']['cameraHorizon']), 30)  # lwj: 绝了！！！就是这个大BUG！气死我了

        self.rotation = 0

        # 初始化相机坐标系
        # 计算原点坐标系到初始坐标点坐标系之间的变换矩阵
        # 第一行求的是初始坐标点坐标系到原点坐标系的变换矩阵(此时camX就是原点)
        # 第二行求逆，即得到原点坐标系到初始坐标点坐标系之间的变换矩阵
        self.invert_pitch = True
        self.camX0_T_origin = self.get_camX0_T_camX(get_camX0_T_origin=True)
        self.camX0_T_origin = utils.safe_inverse_single(self.camX0_T_origin)

        # 初始化地图
        self.map_size = map_size
        self.resolution = args.map_resolution
        if (self.curr_stage == "walkthrough" and args.walkthrough_search == "minViewDistance") or (self.curr_stage == "unshuffle" and args.unshuffle_search == "minViewDistance"):
            self.max_depth = 2  # 4. * 255/25.
        else:
            self.max_depth = 10
        self.selem = skimage.morphology.disk(int(3 * (0.05 / self.resolution)))  #resolution是0.02的时候这里是 2 * 0.02/self.resolution
        self.selem_agent_radius = skimage.morphology.disk(int(3 * (0.05 / self.resolution)))  #although the agent's radius is 0.2m  #resolution是0.02的时候这里是0.06

        self.mapper_dilation = 1
        loc_on_map_size = int(
            np.floor(self.STEP_SIZE / self.resolution / 2))  # +5
        self.loc_on_map_selem = np.ones(
            (loc_on_map_size * 2 + 1, loc_on_map_size * 2 + 1)).astype(bool)
        self.z = [0.05, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
                  1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2]
        self.mapper = Mapper(C, self.position, self.map_size, self.resolution,
                             max_depth=self.max_depth, z_bins=self.z, num_categories=args.num_categories,
                             loc_on_map_selem=self.loc_on_map_selem)

        # 初始化点导航目标
        self.point_goal = None

        # 初始化函数序列
        self._setup_execution()

        # 初始化rgb和depth
        origin_rgb = self.controller.last_event.frame
        origin_depth = self.controller.last_event.depth_frame
        self.rgb = origin_rgb
        self.depth = origin_depth

        self.segHelper = SegmentationHelper(self.controller)

        # 记录模拟器一些环境信息（只用做评估计算metrics）
        if args.use_offline_ai2thor:
            self.reachable_positions_in_simulator = self.controller.reachable_points
        else:
            self.controller.step("GetReachablePositions")
            assert self.controller.last_event.metadata["lastActionSuccess"]
            self.reachable_positions_in_simulator = self.controller.last_event.metadata[
                "actionReturn"]
        self.visited_xz = {self.agent_location_in_simulator_to_tuple[:2]}
        self.visited_xzr = {self.agent_location_in_simulator_to_tuple[:3]}

        objects_metadata = self._get_allobjs_info_in_simulator(args.use_offline_ai2thor)
        self.seen_pickupable_objects = set(
            self._get_pickupable_objects_Id(objects_metadata, visible_only=True))
        self.seen_openable_objects = set(self._get_openable_not_pickupable_objects_Id(
            objects_metadata, visible_only=True))
        self.total_pickupable_or_openable_objects = len(set(
            self._get_pickupable_or_openable_objects_Id(objects_metadata, visible_only=False)))

        # 用于存储planner规划的动作
        self.acts = iter(())
        # 用于存储planner规划的路径
        self.path = None

        # # 记录动作
        # self.acts_history = []

        # 记录失败的动作
        self.obstructed_actions = []

        # 初始化可视化工具
        self.vis = vis
        self.visdom = visdom

        # 记录metrics
        assert self.curr_stage == "walkthrough" or self.curr_stage == "unshuffle"
        if self.curr_stage == "walkthrough":
            self.metrics = {
                f"{self.curr_stage}_actions": [],
                f"{self.curr_stage}_action_successes": [],
                f"{self.curr_stage}/reward": 0,
            }
        else:
            self.metrics = {
                f"{self.curr_stage}_actions": [],
                f"{self.curr_stage}_action_successes": [],
                f"{self.curr_stage}/reward": 0,
                f"pickup_obj_ids":[],
                f"pickup_obj_successes":[],
                f"DropObject_snap_ids":[],
                f"DropObject_snap_successes":[]
            }

        # 如果匹配是基于关系，那么需要维护一个object_tracker
        # self.use_object_tracker = self.solution_config['unshuffle_match'] == 'relation_based'
        # if self.use_object_tracker:
        self.object_tracker = ObjectTrack()

        # 初始化种子
        self.seed = 0

        #初始化mass的探索策略
        if (self.curr_stage == "walkthrough" and args.walkthrough_search == "mass") or (self.curr_stage == "unshuffle" and args.unshuffle_search == "mass"):

            self.mass_exploration_modal = nn.Sequential(
                nn.Conv2d(54, 64, 3, padding=1),

                nn.GroupNorm(1, 64),
                nn.ReLU(),

                nn.Conv2d(64, 64, 3, padding=1),

                nn.GroupNorm(1, 64),
                nn.ReLU(),

                nn.Conv2d(64, 64, 3, padding=1),

                nn.GroupNorm(1, 64),
                nn.ReLU(),

                nn.Conv2d(64, 64, 3, padding=1),

                nn.GroupNorm(1, 64),
                nn.ReLU(),

                nn.Conv2d(64, 1, 3, padding=1),
                )

            self.mass_exploration_modal.load_state_dict(torch.load(args.mass_modal_path))
            self.mass_exploration_modal.eval()
            self.mass_exploration_modal.to(torch.device(f"cuda:{args.mass_device}"))

    # 计算机器人当前坐标系到初始坐标点坐标系之间的变换矩阵

    def get_camX0_T_camX(self, get_camX0_T_origin=False):
        '''
        Get transformation matrix between first position (camX0) and current position (camX)
        '''
        position = np.array(list(self.position.values()))

        # in aithor negative pitch is up - turn this on if need the reverse
        head_tilt = self.head_tilt
        if self.invert_pitch:
            head_tilt = -head_tilt

        rx = np.radians(head_tilt)  # pitch
        rotation = self.rotation
        if rotation >= 180:
            rotation = rotation - 360
        if rotation < -180:
            rotation = 360 + rotation
        ry = np.radians(rotation)  # yaw
        rz = 0.  # roll is always 0
        rotm = eul2rotm_py(np.array([rx]), np.array(
            [ry]), np.array([rz]))  # 旋转矩阵
        origin_T_camX = np.eye(4)
        origin_T_camX[0:3, 0:3] = rotm
        origin_T_camX[0:3, 3] = position  # 变换矩阵
        origin_T_camX = torch.from_numpy(origin_T_camX)
        if get_camX0_T_origin:
            camX0_T_camX = origin_T_camX
        else:
            camX0_T_camX = torch.matmul(self.camX0_T_origin, origin_T_camX)
        return camX0_T_camX

    def _setup_execution(self):
        self.execution = LifoQueue(maxsize=200)
        fn = lambda: self._cover_fn()
        self.execution.put(fn)
        fn = lambda: self._init_fn()
        self.execution.put(fn)

    def _init_fn(self):
        self.init_on = True
        # 原先是 低头2次+旋转4次+抬头2次 （8，见到了理论上1+1+4个不同图像）
        # 若是 旋转4次+低头2次+旋转4次+抬头2次 （12，见到了理论上4+1+4个不同图像）
        # 若是 低头2次+旋转+抬头2次+旋转+低头2次+旋转+抬头2次+旋转（12，见到了理论上4+4+4个不同图像）
        for i in range(int(360 / self.DT)):
            if i % 2 == 0:
                for j in range(int(60 / self.HORIZON_DT)):
                    yield LOOK_DOWN
            else:
                for j in range(int(60 / self.HORIZON_DT)):
                    yield LOOK_UP

            yield ROTATE_LEFT

            self.init_on = False

    def _sample_point_with_mass_network(self):

        occupancy_map = self.mapper.map
        semantic_map = (self.mapper.semantic_map[:,:,:,:54] > 0).astype(int)
        semantic_map[:,:,:,0] = occupancy_map
        semantic_map_tor = torch.from_numpy(semantic_map).to(torch.device(f"cuda:{args.mass_device}"))
        semantic_map_tor = semantic_map_tor.type(torch.cuda.FloatTensor)

        map_size = occupancy_map.shape[0]
        prediction = self.mass_exploration_modal(semantic_map_tor.sum(dim=2).unsqueeze(0).permute(0, 3, 1, 2))
        prediction = functional.softmax(prediction.view(map_size * map_size), dim=0)
        goal = int(torch.multinomial(prediction, 1))
        return goal // map_size, goal % map_size

    def _cover_fn(self) -> None:
        '''检查未探索区域是否小于阈值(20)，如果大于阈值，生成下一个探索点'''
        unexplored = self._get_unexplored()
        if args.debug_print:
            print("Unexplored", np.sum(unexplored))
        explored = np.sum(unexplored) < 20
        if explored:
            print('Unexplored area < 20. Exploration finished')
        else:
            if (self.curr_stage == "walkthrough" and args.walkthrough_search == "mass") or (self.curr_stage == "unshuffle" and args.unshuffle_search == "mass"):
                ind_i,ind_j = self._sample_point_with_mass_network()
            else:
                ind_i, ind_j = self._sample_point_in_unexplored_reachable(unexplored)
            
            if ind_i is None or ind_j is None:
                return
            
            self.point_goal = [ind_i, ind_j]
            fn = lambda: self._cover_fn()
            self.execution.put(fn)
            fn = lambda: self._point_goal_fn_assigned(
                np.array([ind_j, ind_i]), explore_mode=True, dist_thresh=0.5)
            self.execution.put(fn)

    # ? lwj:  逻辑可以优化，这里的逻辑：map记录过点的地方就不会被当作unexplored了，但是map记录过点的地方只是add_observation加进去的，可能由于depth太远有误差，是可以再走近探索的
    def _get_unexplored(self):
        # 机器人去过or室内记录的点小于100个的(根据地图机器人能去的地方，墙外不能去)
        reachable = self._get_reachable_area()
        explored_point_count = 1
        explored = self.mapper.get_explored_map(
            self.selem, explored_point_count)  # 机器人去过or map记录过点的(即视角看见过的地方)
        unexplored = np.invert(explored)  # 机器人没去过and没有记录过点的
        # 机器人没去过and 没有记录过点的 and 可以到达的
        unexplored1 = np.logical_and(unexplored, reachable)
        # added to remove noise effects
        disk = skimage.morphology.disk(2)
        unexplored2 = skimage.morphology.binary_opening(unexplored1, disk)
        self.unexplored_area = np.sum(unexplored2)

        return unexplored2

    # 根据地图从当前位置能去的地方
    def _get_reachable_area(self):
        traversible = self.mapper.get_traversible_map(
            self.selem_agent_radius, POINT_COUNT, loc_on_map_traversible=True)  # 机器人去过的地方 or map点小于100的地方 (不考虑bound)
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        if args.walkthrough_search == "cover" or args.walkthrough_search == "minViewDistance" or args.walkthrough_search == 'cover_nearest':
            step_pix = int(args.STEP_SIZE / args.map_resolution)
            pooled_map = pool2d(traversible,kernel_size=step_pix,stride=step_pix,pool_mode='avg')
            indices = np.indices(traversible.shape) // step_pix
            traversible = pooled_map[indices[0], indices[1]]
            traversible = traversible > 0.8
            #让机器人所处的大格子都是True
            index = state_xy - state_xy % step_pix
            traversible[index[1]:index[1]+step_pix,index[0]:index[0]+step_pix] = True
        planner = FMMPlanner(traversible, 360 // self.DT, int(self.STEP_SIZE /
                             self.mapper.resolution), self.obstructed_actions, self.visdom)
    
        state_theta = self.mapper.get_rotation_on_map() + np.pi / 2
        reachable = planner.set_goal(state_xy)  # dd_mask
        return reachable

    def _sample_point_in_unexplored_reachable(self, unexplored):
        # Given the map, sample a point randomly in the open space.est

        if (self.curr_stage == "walkthrough" and (args.walkthrough_search == "cover_nearest" )) or (self.curr_stage == "unshuffle" and (args.unshuffle_search == "cover_nearest")) :
            unexplored_indexs = np.transpose(np.where(unexplored))
            state_xy = self.mapper.get_position_on_map()
            dist = distance.cdist(np.expand_dims(np.array([int(state_xy[1]),int(state_xy[0])]), axis=0), unexplored_indexs)[0]
            #筛选0.5m以外的最近的点
            dist_meter = dist * args.map_resolution
            without_thresh = dist_meter > 0.75
            dist = dist[without_thresh]
            unexplored_indexs = unexplored_indexs[without_thresh]

            sorted_dist_indices = np.argsort(dist)

            dist = dist[sorted_dist_indices]
            unexplored_sorted_points = unexplored_indexs[sorted_dist_indices]
            if len(unexplored_sorted_points) == 0:
                return None, None
        
            unexplored_sorted_points = unexplored_sorted_points[0:1000]
            dist = dist[0:1000]

            # 选择最大高度最小的点
            topdown_highest_obstacle_map = self.mapper.get_topdown_highest_obstacle_map()

            unexplored_max_heights = []
            for coords in unexplored_sorted_points:
                max_height = get_max_height_on_line(topdown_highest_obstacle_map, coords, np.array([int(state_xy[1]),int(state_xy[0])]))
                unexplored_max_heights.append(max_height)
            # 取排序后前三个元素的索引
            # sorted_max_height_indices = np.lexsort((unexplored_max_heights, np.arange(len(unexplored_max_heights))))
            sorted_max_height_indices = np.lexsort((dist, unexplored_max_heights))

            min_indices = sorted_max_height_indices[:3]
            # 从前三个元素中随机选一个
            rng = np.random.RandomState(self.seed)
            self.seed += 1
        
            min_indice = rng.choice(min_indices)
        
            min_height_coord = unexplored_sorted_points[min_indice]
            return min_height_coord[0], min_height_coord[1]

        ind_i, ind_j = np.where(unexplored)
        rng = np.random.RandomState(self.seed)
        self.seed += 1
        ind = rng.randint(ind_i.shape[0])
        return ind_i[ind], ind_j[ind]

    def _point_goal_fn_assigned(self, goal_loc_cell: np.ndarray, explore_mode: bool, dist_thresh: float, held_mode: bool = False, iters=20):
        '''点导航函数
        goal_loc_cell:目标点坐标
        explore_mode:探索模式
        dist_thresh:判断是否到达目标点附近的阈值
        iters:允许重复规划的次数
        '''
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        state_theta = self.mapper.get_rotation_on_map() + np.pi / 2
        reached = self._check_point_goal_reached(goal_loc_cell, dist_thresh)
        if reached:  # or dists_equal:
            if args.debug_print:
                print("REACHED")
            # 到达目的地后低头原地旋转一圈
            if explore_mode and args.walkthrough_search == 'cover' and args.unshuffle_search == 'cover':
                yield LOOK_DOWN
                yield LOOK_DOWN
                for _ in range(360 // self.DT):
                    yield ROTATE_LEFT
                yield LOOK_UP
                yield LOOK_UP
            return
        else:
            if iters == 0:
                return
            traversible = self.mapper.get_traversible_map(
                self.selem_agent_radius, POINT_COUNT, loc_on_map_traversible=True)
            planner = FMMPlanner(traversible, 360 // self.DT, int(
                self.STEP_SIZE / self.mapper.resolution), self.obstructed_actions, self.visdom)
            goal_loc_cell = goal_loc_cell.astype(np.int32)
            reachable = planner.set_goal(goal_loc_cell)
            if args.debug_print:
                print("goal_loc_cell", goal_loc_cell)
                print("reachable[state_xy[1], state_xy[0]]", reachable[state_xy[1], state_xy[0]])
                # if not reachable[state_xy[1], state_xy[0]]:
                #     print('debug here !!!!!!!!')
            if reachable[state_xy[1], state_xy[0]]:
                # a, state, act_seq = planner.get_action(np.array([state_xy[0], state_xy[1], state_theta]))
                act_seq, path = planner.get_action_sequence_dij(state_xy, self.rotation, goal_loc_cell, held_mode)
            else:
                a, state, act_seq = planner.get_action(np.array([state_xy[0], state_xy[1], state_theta]))
                path = None
            self.act_seq = act_seq
            self.path = path
            if explore_mode:
                for a in act_seq[:10]:
                    yield a
            else:
                for a in act_seq:
                    yield a
            # 执行完act_seq后，再在execution中加一个点导航函数用于检查目标点是否到达
            fn = lambda: self._point_goal_fn_assigned(
                goal_loc_cell, explore_mode=explore_mode, dist_thresh=dist_thresh, iters=iters - 1, held_mode=held_mode)
            self.execution.put(fn)

    # dist_thresh = 0.5是现实世界中的距离（m)
    def _check_point_goal_reached(self, goal_loc_cell, dist_thresh=0.5):
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        dist = np.sqrt(np.sum(np.square(state_xy - goal_loc_cell))
                       ) * self.mapper.resolution
        topdown_highest_obstacle_map = self.mapper.get_topdown_highest_obstacle_map()

        max_height = get_max_height_on_line(topdown_highest_obstacle_map, np.array(
            [goal_loc_cell[1], goal_loc_cell[0]]), np.array([state_xy[1], state_xy[0]]))

        return dist < dist_thresh and max_height < 16

    def check_successful_action(self, rgb: np.ndarray = None, rgb_prev: np.ndarray = None, perc_diff_thresh: float = None) -> bool:
        '''根据动作前后rgb图片的差别判断动作是否成功'''
        return self.controller.last_event.metadata['lastActionSuccess']
        num_diff = np.sum(np.sum(rgb_prev.reshape(
            self.W * self.H, 3) - rgb.reshape(self.W * self.H, 3), 1) > 0)

        if num_diff < perc_diff_thresh * self.W * self.H:
            success = False
        else:
            success = True
        # self.rgb_prev = rgb
        return success

    def _get_agent_location_in_simulator(self):
        metadata = self.controller.last_event.metadata
        return {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"],
            "standing": metadata.get("isStanding", metadata["agent"].get("isStanding")),
        }

    # get xzrsh
    @property
    def agent_location_in_simulator_to_tuple(self):
        agent_loc = self._get_agent_location_in_simulator()
        return (
            round(agent_loc["x"], 2),
            round(agent_loc["z"], 2),
            round_to_factor(agent_loc["rotation"], 90) % 360,
            1 * agent_loc["standing"],
            round_to_factor(agent_loc["horizon"], 30) % 360,
        )

    def _get_allobjs_info_in_simulator(self, use_offline_ai2thor):
        if use_offline_ai2thor:
            objects_metadata = self.controller.all_objects()
        else:
            objects_metadata = self.controller.last_event.metadata['objects']
        assert len(objects_metadata) != 0, f"objects metadata == 0, please check!"
        return objects_metadata

    def _get_pickupable_objects_Id(self, objects_metadata, visible_only=False):
        return [
            o['objectId'] for o in objects_metadata
            if (o['visible'] or not visible_only) and o['pickupable']
        ]

    def _get_openable_not_pickupable_objects_Id(self, objects_metadata, visible_only=False):
        return [
            o['objectId'] for o in objects_metadata
            if (o['visible'] or not visible_only) and (o['openable'] and not o['pickupable'])
        ]

    def _get_pickupable_or_openable_objects_Id(self, objects_metadata, visible_only=False):
        return [
            o['objectId'] for o in objects_metadata
            if (o['visible'] or not visible_only) and (o['openable'] or o['pickupable'])
        ]

    def _update_metrics(self, action_name, action_success, paras = dict(),calReward = False):
        self.metrics[f"{self.curr_stage}_actions"].append(action_name)
        self.metrics[f"{self.curr_stage}_action_successes"].append(action_success)
        if(action_name == 'PickupObject'):
            self.metrics[f"pickup_obj_ids"].append(paras["objectId"])
            self.metrics[f"pickup_obj_successes"].append(action_success)
        if(action_name == 'DropObject_snap'):
            self.metrics[f"DropObject_snap_ids"].append(paras["objectId"])
            self.metrics[f"DropObject_snap_successes"].append(action_success)


        if self.curr_stage == 'walkthrough':
            total_seen_before = len(self.seen_pickupable_objects) + len(self.seen_openable_objects)
            prop_seen_before = total_seen_before / self.total_pickupable_or_openable_objects

            # Updating (recorded) visited locations in simulator (only for metrics calculation)
            agent_loc_tuple = self.agent_location_in_simulator_to_tuple
            self.visited_xz.add(agent_loc_tuple[:2])
            self.visited_xzr.add(agent_loc_tuple[:3])

            objects_metadata = self._get_allobjs_info_in_simulator(use_offline_ai2thor=args.use_offline_ai2thor)
            # Updating seen openable
            for objId in self._get_openable_not_pickupable_objects_Id(objects_metadata, visible_only=True):
                if objId not in self.seen_openable_objects:
                    self.seen_openable_objects.add(objId)

            # Updating seen pickupable
            for objId in self._get_pickupable_objects_Id(objects_metadata, visible_only=True):
                if objId not in self.seen_pickupable_objects:
                    self.seen_pickupable_objects.add(objId)

            total_seen_after = len(
                self.seen_pickupable_objects) + len(self.seen_openable_objects)
            prop_seen_after = total_seen_after / self.total_pickupable_or_openable_objects
            if calReward:
                reward = 5 * (prop_seen_after - prop_seen_before)
                if action_name == 'Pass' and prop_seen_after > 0.5:
                    reward += 5 * (prop_seen_after + (prop_seen_after > 0.98))

        elif self.curr_stage == 'unshuffle':
            if calReward:
            # if action_name in ['PickupObject','OpenObject','CloseObject','PutObject','DropObject']:
                curr_energies_dict = self.get_curr_pose_difference_energy(self.curr_stage)
                curr_energies = np.array(list(curr_energies_dict.values()))
                curr_energy = curr_energies.sum()
                # changed_objs_id = [k for k, v in curr_energies_dict.items() if v > 0]
                # changed_objs_parent_goal = [self.walkthrough_objs_id_to_pose[id]['parentReceptacles'] for id in changed_objs_id]
                # changed_objs_parent_curr = [curr_objs_id_to_pose[id]['parentReceptacles'] for id in changed_objs_id]
                # changed = [f"{changed_objs_id[i]}:from {changed_objs_parent_goal[i]} --> {changed_objs_parent_curr[i]}" for i in range(len(changed_objs_id))]

                energy_change = self.last_energy - curr_energy
                self.last_energy = curr_energy
                reward = energy_change
            # else:
            #     reward = 0
        if calReward:
            self.metrics[f"{self.curr_stage}/reward"] += reward
            if args.debug_print or self.step_from_stage_start % 10 == 0:
                print(f'Process{self.process_id}-{self.curr_stage} Step {self.step_from_stage_start}: {action_name}, --(reward: {reward})')
        else:
            if args.debug_print or self.step_from_stage_start % 10 == 0:
                print(f'Process{self.process_id}-{self.curr_stage} Step {self.step_from_stage_start}: {action_name}')


    def get_current_objs_id_to_pose(self, stage = None):
        objs_id_to_pose = dict()
        if stage == 'rearrange':
            objects_metadata = self._get_allobjs_info_in_simulator(use_offline_ai2thor=False)
        else:
            objects_metadata = self._get_allobjs_info_in_simulator(use_offline_ai2thor=args.use_offline_ai2thor)
        for obj in objects_metadata:
            # 排除一些结构物: Floor, Wall, Door等等
            if "Cracked" in obj["objectId"]:
                continue
            if obj['openable'] or obj.get('objectOrientedBoundingBox') is not None:
                if 'Cracked' in obj['objectId']:
                    continue
                assert obj['objectId'] not in objs_id_to_pose
                objs_id_to_pose[obj['objectId']] = obj

        if self.curr_stage == 'unshuffle':
            for objId_walkthrough in self.walkthrough_objs_id_to_pose.keys():
                if objId_walkthrough not in objs_id_to_pose.keys():
                   # assume the disappeared objects are broken
                    objs_id_to_pose[objId_walkthrough] = {
                        **self.walkthrough_objs_id_to_pose[objId_walkthrough],
                        "isBroken": True,
                        'broken': True,
                        "position": None,
                        "rotation": None,
                        "openness": None,
                    }
            assert len(self.walkthrough_objs_id_to_pose.keys()) == len(objs_id_to_pose.keys()), \
                f"obj poses dismatch ! walkthrough - unshuffle = {set(self.walkthrough_objs_id_to_pose.keys() - set(objs_id_to_pose.keys()))}, \
            unshuffle - walkthrough = {set(objs_id_to_pose.keys()) - set(self.walkthrough_objs_id_to_pose.keys())}"

            if self.step_from_stage_start == 0:
                # If we find a broken goal object, we will simply pretend as though it was not
                # broken. This means the agent can never succeed in unshuffling, this means it is
                # possible that even a perfect agent will not succeed for some tasks.
                broken_objs_id_in_walkthrough = [
                    objId for objId, obj in self.walkthrough_objs_id_to_pose.items() if obj['isBroken']]
                for broken_objId in broken_objs_id_in_walkthrough:
                    self.walkthrough_objs_id_to_pose[broken_objId]["isBroken"] = False
                    objs_id_to_pose[broken_objId]["isBroken"] = False

        return objs_id_to_pose

    def update_position_and_rotation(self, act_id: int) -> None:
        '''根据动作更新Agent(在地图坐标系)的位置和角度
           x轴向右
           z轴向前
           y轴向上
           rotation是和z轴的夹角
           head_tilt是和y轴的夹角，向下为正
        '''
        if 'Rotate' in self.act_id_to_name[act_id]:
            if 'Left' in self.act_id_to_name[act_id]:
                self.rotation -= self.DT
            else:
                self.rotation += self.DT
            self.rotation %= 360
        elif 'Move' in self.act_id_to_name[act_id]:
            if act_id == MOVE_AHEAD:
                self.position['x'] += np.sin(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
                self.position['z'] += np.cos(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
            elif act_id == MOVE_BACK:
                self.position['x'] -= np.sin(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
                self.position['z'] -= np.cos(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
            elif act_id == MOVE_LEFT:
                self.position['x'] -= np.cos(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
                self.position['z'] += np.sin(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
            elif act_id == MOVE_RIGHT:
                self.position['x'] += np.cos(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
                self.position['z'] -= np.sin(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
        elif 'Look' in self.act_id_to_name[act_id]:
            if 'Down' in self.act_id_to_name[act_id]:
                self.head_tilt += self.HORIZON_DT
            else:
                self.head_tilt -= self.HORIZON_DT

    def reset_execution_and_acts(self, act_id: int) -> None:
        '''
        重新规划栈中的函数序列self.execution 
        和已经规划的动作序列 self.acts
        '''
        # 清空函数栈和动作
        self.execution = LifoQueue(maxsize=200)
        self.acts = None
        fn = lambda: self._cover_fn()
        self.execution.put(fn)
        # 重新选一个点导航
        if(self.point_goal == None):
            return
        self.point_goal = self.choose_reachable_map_pos_in_same_room(
            self.point_goal, thresh=0.5, explore_mode=True)
        if self.point_goal is None:
            return 
        ind_i, ind_j = self.point_goal
        fn = lambda: self._point_goal_fn_assigned(goal_loc_cell=np.array(
            [ind_j, ind_i]), dist_thresh=0.5, explore_mode=True)
        self.execution.put(fn)
        # 失败后随机移动
        fn = lambda: self._random_move_fn(act_id,held_mode=False)
        self.execution.put(fn)

    def _increment_num_steps_taken(self):
        self.step_from_stage_start += 1

    # lwj: : 一个改进思路，这里先把dist为0（或小于一定阈值的）删掉，即选原point_goal附近的点，而不是仍然保持不变
    def choose_reachable_map_pos_in_same_room(self, map_pos, thresh=1, within_thresh=False, explore_mode = False):
        '''随机选取在map_pos 1.5m以内并且可达的点'''

        if args.debug_print:
            print("choose_reachable_map_pos_in_same_room")
        reachable = self._get_reachable_area()
        step_pix = int(args.STEP_SIZE / args.map_resolution)

        map_pos = [int(map_pos[0]), int(map_pos[1])]
        # 下采样后的地图
        pooled_reachable = pool2d(reachable, kernel_size=step_pix, stride=step_pix, pool_mode='avg')

        pooled_reachable = pooled_reachable == 1
        # 求所有在pooled_map上reachable的点的原地图坐标
        original_reachable_indexs = np.transpose(np.where(reachable))
        pooled_reachable_indexs = (original_reachable_indexs / step_pix).astype(int)
        valid_indices = np.logical_and(pooled_reachable_indexs[:, 0] < pooled_reachable.shape[0], pooled_reachable_indexs[:, 1] < pooled_reachable.shape[1])

     
        final_indexs = original_reachable_indexs[valid_indices][pooled_reachable[pooled_reachable_indexs[valid_indices, 0], pooled_reachable_indexs[valid_indices, 1]] == 1]
        # print("final_index.size", final_indexs.shape)
        # 距离列表
        dist = distance.cdist(np.expand_dims(map_pos, axis=0), final_indexs)[0]
        sorted_dist_indices = np.argsort(dist)

        dist = dist[sorted_dist_indices]
        reachable_sorted_points = final_indexs[sorted_dist_indices]

        if explore_mode:
            dist_outof_thresh = (dist * args.map_resolution > thresh)
                # 所有以内的可达点，并且按距离排序了
            reachable_sorted_points_outof_thresh = reachable_sorted_points[dist_outof_thresh]
        # if explore_mode and len(reachable_sorted_points_outof_thresh):
            reachable_sorted_points = reachable_sorted_points_outof_thresh
            dist = dist[dist_outof_thresh]

        if not len(reachable_sorted_points):
            # 这个地方lyy能不能让它返回最近的
            return None
        # reachable_sorted_points_2 = reachable_sorted_points[1000:2000]
        reachable_sorted_points = reachable_sorted_points[0:1000]
        dist = dist[0:1000]
        
        # 选择最大高度最小的点
        topdown_highest_obstacle_map = self.mapper.get_topdown_highest_obstacle_map()

        reachable_max_heights = []
        for coords in reachable_sorted_points:
            max_height = get_max_height_on_line(topdown_highest_obstacle_map, coords, map_pos)
            reachable_max_heights.append(max_height)
        
        # 取排序后前三个元素的索引
        # sorted_max_height_indices = np.lexsort((reachable_max_heights, np.arange(len(reachable_max_heights))))
        sorted_max_height_indices = np.lexsort((dist, reachable_max_heights))
        # sorted_max_height_indices = np.argsort(reachable_max_heights)
        #如果有1.5m内的点，选10个；没有的话则选最近的3个
        min_indices = sorted_max_height_indices[:3]
        # 从前三个元素中随机选一个
        rng = np.random.RandomState(self.seed)
        self.seed += 1
        
        if len(min_indices) == 0:
            print('debug here')
        min_indice = rng.choice(min_indices)
        # print("reachable_max_heights", reachable_max_heights)
        # print("rng",rng)
        # print("seed",self.seed)
        # print("min_indices",min_indices)
        # print("min_indice",min_indice)
        
        min_height_coord = reachable_sorted_points[min_indice]
        return min_height_coord[0], min_height_coord[1]

    def _random_move_fn(self, act_id,held_mode):
        '''尝试小范围移动'''
    
        # yield LOOK_DOWN
        # yield LOOK_DOWN
        # reachable = self._get_reachable_area()
        # step_pix = int(args.STEP_SIZE / args.map_resolution)
        # 下采样后的地图
        # pooled_reachable = pool2d(reachable, kernel_size=step_pix, stride=step_pix, pool_mode='avg')
        # state_xy = self.mapper.get_position_on_map()
        # shape_y = self.mapper.map_sz
        # state_x, state_y = state_xy
        # pooled_state_x, pooled_state_y = int(state_xy / step_pix)
        # if not held_mode:
        #     yield LOOK_DOWN
        #     yield LOOK_DOWN
        #     yield LOOK_UP
        #     yield LOOK_UP
        if act_id == MOVE_AHEAD:
            # if reachable[max(state_y - step_pix, 0): state_y, max(state_x - step_pix, 0): state_x].sum() > \
            #     reachable[max(state_y - step_pix, 0): state_y, state_x: min(state_x + step_pix, shape_x)].sum():
            yield MOVE_LEFT
            # else:
            #     yield MOVE_LEFT
        elif act_id == MOVE_LEFT:
            yield MOVE_BACK
        elif act_id == MOVE_BACK:
            yield MOVE_RIGHT
        else:
            yield MOVE_AHEAD

    # ？？？？lwj:  不太懂为什么是这种添加方式？
    def update_obstructed_actions(self, act_id: int) -> None:
        prev_len = len(self.obstructed_actions)
        if prev_len > 4000:
            pass
        else:
            for idx in range(prev_len):
                obstructed_acts = self.obstructed_actions[idx]
                self.obstructed_actions.append(obstructed_acts + [act_id])
            self.obstructed_actions.append([act_id])

    def explore_env(self):
        step = 0  # 仅仅表示此次探索的step
        prev_act_id = 0
        num_sampled = 0
        if self.curr_stage == 'walkthrough':
            explore_policy = self.solution_config['walkthrough_search']
        else:
            explore_policy = self.solution_config['unshuffle_search']
        if args.debug_print:
            print("explore_policy", explore_policy)
        while step < self.step_max and (explore_policy == 'cover_continue' or prev_act_id != DONE):
            # 1. 获取rgb和depth
            rgb_prev = self.rgb
            depth_prev = self.depth
            if args.use_seg:
                # seg_prev: H * W * (num_category + 1), segmented_dict:{'scores':, 'categories', 'masks'} 
                seg_prev, segmented_dict = self.segHelper.get_seg_pred(rgb_prev)
            else:
                seg_prev, segmented_dict = None, None

            # seg_prev_vis = visualize_segmentationRGB(rgb_prev, segmented_dict, visualize_sem_seg=True)
            # visdomImage([rgb_prev, seg_prev_vis], self.visdom, tag='subplot', info=['rgb', 'rgb_seg'],
                        #  max_depth=self.max_depth, prev_action=self.act_id_to_name[prev_act_id], current_step=self.step_from_stage_start)

            # 2. 添加到地图中
            object_track_dict, curr_judge_new_and_centriod = self.mapper.add_observation(self.position,
                                        self.rotation,
                                        -self.head_tilt,
                                        depth_prev,
                                        seg_prev,
                                        segmented_dict,
                                        self.object_tracker.get_objects_track_dict(),
                                        add_obs=True,
                                        add_seg=args.use_seg)
            self.object_tracker.set_objects_track_dict(object_track_dict)

            # if self.use_object_tracker:
            # if self.solution_config['unshuffle_match'] == 'relation_based':   #lyy注意一下这里
            #     self.object_tracker.update_by_category_and_distance(mapper = self.mapper , depth=depth_prev, segmented_dict=segmented_dict)
            # elif self.solution_config['unshuffle_match'] == 'instance_based' or self.solution_config['unshuffle_match'] == 'map_based':
            self.object_tracker.update_by_map_and_instance_and_feature(segmented_dict, curr_judge_new_and_centriod)

            # # # # #用于测试可视化效果【用于debug】
            # seg_prev_vis = visualize_segmentationRGB(rgb_prev, segmented_dict, visualize_sem_seg=True)
            # traversible_map =  np.invert(self.mapper.get_traversible_map(self.selem_agent_radius, 1,loc_on_map_traversible=True))
            # # # img_local_map_sum = self.mapper.get_local_map_sum()
            
            # map_topdown_seg = self.mapper.get_topdown_semantic_map()
            # explored = self.mapper.get_explored_map(self.selem, point_count=1) #机器人去过的地方 or map有点的地方
            # obstacle = self.mapper.get_obstacle()    #map上点的数量大于100的记为障碍物
            # visited = self.mapper.get_visited()
            # map_for_distance = self.mapper.get_map_for_view_distance()
            
            # map_topdown_seg_vis = visualize_topdownSemanticMap(map_topdown_seg, explored, obstacle, visited)
            # # map_topdown_highest_obstacle = self.mapper.get_topdown_highest_obstacle_map()
            # test_obj_id = 'Box|2|2'
            # if test_obj_id in self.controller.last_event.instance_masks:
            #     seg_test_obj = self.controller.last_event.instance_masks[test_obj_id]
            # else:
            #     seg_test_obj = np.zeros_like(rgb_prev)
            # map_seg_test_obj = self.mapper.get_hierarchical_semantic_map(category_id=35)

            # visdomImage([rgb_prev, seg_prev_vis, seg_test_obj, map_seg_test_obj, traversible_map, map_topdown_seg_vis], self.visdom, tag='subplot', info=['rgb', 'rgb_seg','seg_mask', 'map_seg_hierarchical', 'map_bool', 'map_seg_topdown'],
            #              max_depth=self.max_depth, prev_action=self.act_id_to_name[prev_act_id], current_step=self.step_from_stage_start, map_for_distance = map_for_distance)

            # if step > 13:
            #     print('debug image')
            #     visdomImage([rgb_prev, depth_prev, seg_SideTable, traversible_map, map_topdown_seg_vis, map_seg_SideTable], self.visdom, tag='subplot', info=['rgb', 'depth', 'seg_mask', 'map_bool', 'map_seg_topdown', 'map_seg_hierarchical'],
            #                 max_depth=self.max_depth, prev_action=self.act_id_to_name[prev_act_id], current_step=self.step_from_stage_start)
            # visdomImage([rgb_prev, depth_prev, seg_prev_vis, traversible_map, map_topdown_seg_vis, map_topdown_highest_obstacle],
            #             self.visdom, tag='subplot', info=['rgb', 'depth', 'rgb_seg', 'map_bool', 'map_seg_topdown', 'map_high_obs_topdown'],
            #              max_depth=self.max_depth, prev_action=self.act_id_to_name[prev_act_id], current_step=self.step_from_stage_start)
            # pdb.set_trace()

            # 3. 根据地图选择动作
            if self.acts == None:
                act_id = None
            else:
                act_id = next(self.acts, None)
            if act_id is None:
                while self.execution.qsize() > 0:
                    op = self.execution.get()
                    # point_goal_fn里面有yield，所以返回一个生成器，函数并没有真正执行，只有用next()方法时函数才执行
                    self.acts = op()
                    if self.acts is not None:
                        act_id = next(self.acts, None)
                        if act_id is not None:
                            break
            if act_id is None:
                act_id = DONE
            prev_act_id = act_id
            # print("Step ", self.step_from_stage_start, self.act_id_to_name[act_id])
            # 4. 如果有需要的话进行可视化，保存第一视角视频和地图
            if self.vis != None:
                if args.use_seg:
                    seg_prev_vis = visualize_segmentationRGB(rgb_prev, segmented_dict, visualize_sem_seg=True)
                else:
                    seg_prev_vis = None

                if self.step_from_stage_start == 0:
                    self.vis.add_frame(image=rgb_prev, depth=depth_prev, seg_image=seg_prev_vis, add_map=True, add_map_seg=True, mapper=self.mapper,
                                       point_goal=self.point_goal, selem=self.selem, selem_agent_radius=self.selem_agent_radius, path=self.path, step=self.step_from_stage_start,
                                       stage=self.curr_stage, visdom=self.visdom, action=self.act_id_to_name[act_id])
                else:
                    self.vis.add_frame(image=rgb_prev, depth=depth_prev, seg_image=seg_prev_vis, add_map=True, add_map_seg=True, mapper=self.mapper,
                                       point_goal=self.point_goal, selem=self.selem, selem_agent_radius=self.selem_agent_radius, path=self.path, step=self.step_from_stage_start,
                                       stage=self.curr_stage, visdom=self.visdom, action=self.act_id_to_name[act_id])
            # 5. 执行动作
            act_name = self.act_id_to_name[act_id]
            self.controller.step(action=act_name)
            
            self.rgb = self.controller.last_event.frame
            self.depth = self.controller.last_event.depth_frame
            # pdb.set_trace()

            # 6. 如果动作成功，更新机器人的position和rotation
            act_isSuccess = self.check_successful_action(
                rgb=self.rgb, rgb_prev=rgb_prev, perc_diff_thresh=0.05) == True
            if act_isSuccess:
                self.update_position_and_rotation(act_id)
            # 如果动作失败，重新规划函数序列和已经规划的动作序列
            elif self.init_on == False:
                if 'Move' in self.act_id_to_name[act_id]:
                    self.mapper.add_obstacle_in_front_of_agent(
                        act_name=act_name, rotation=self.rotation)
                    # pass # lyy!
                # self.update_obstructed_actions(act_id)  #lyy!好像这个只用于FMM原始的规划
                self.reset_execution_and_acts(act_id)
                if args.debug_print:
                    print("ACTION FAILED.", self.controller.last_event.metadata['errorMessage'])

            # 7. 在探索策略为cover_continue时，pass后再随机选取10个点进行点导航
            if explore_policy == 'cover_continue' and act_id == DONE:
                if args.debug_print:
                    print("num_sampled", num_sampled)
                num_sampled += 1
                if num_sampled > 10:
                    break
                ind_i, ind_j = self._sample_point_in_reachable_reachable()
                if not ind_i:
                    break
                # fn = lambda: self._cover_fn()
                # self.execution.put(fn)
                self.point_goal = ind_i, ind_j
                fn = lambda: self._point_goal_fn_assigned(goal_loc_cell=np.array(
                    [ind_j, ind_i]), dist_thresh=0.5, explore_mode=False)
                self.execution.put(fn)


            step += 1
            self._increment_num_steps_taken()
            self._update_metrics(action_name=act_name, action_success=act_isSuccess, calReward=False)

            # if step % 10 == 0:
            
    def move_all_maps_to_center(self):
        pass

    def _sample_point_in_reachable_reachable(self):
        reachable = self._get_reachable_area()
        state_xy = self.mapper.get_position_on_map()

        inds_i, inds_j = np.where(reachable)
        dist = np.sqrt(np.sum(np.square(np.expand_dims(
            state_xy, axis=1) - np.stack([inds_i, inds_j], axis=0)), axis=0))
        dist_thresh = dist > 20.0
        inds_i = inds_i[dist_thresh]
        inds_j = inds_j[dist_thresh]
        if inds_i.shape[0] == 0:
            print("FOUND NO REACHABLE INDICES")
            return [], []
        ind = np.random.randint(inds_i.shape[0])
        ind_i, ind_j = inds_i[ind], inds_j[ind]
        return ind_i, ind_j

    def get_curr_pose_difference_energy(self, stage):
        curr_objs_id_to_pose = self.get_current_objs_id_to_pose(stage=stage)
        curr_energies_dict = pose_difference_energy(
            goal_poses=self.walkthrough_objs_id_to_pose, cur_poses=curr_objs_id_to_pose)
        return curr_energies_dict

    def get_metrics(self):
        # 探索完成后, 计算metrics
        if self.curr_stage == 'walkthrough':
            n_reachable = len(self.reachable_positions_in_simulator)
            n_obj_seen = len(self.seen_openable_objects) + \
                len(self.seen_pickupable_objects)
            self.metrics = {
                **self.metrics,
                **{
                    f'{self.curr_stage}/ep_length': self.step_from_stage_start,
                    f'{self.curr_stage}/num_explored_xz': len(self.visited_xz),
                    f'{self.curr_stage}/num_explored_xzr': len(self.visited_xzr),
                    f'{self.curr_stage}/prop_visited_xz': len(self.visited_xz) / n_reachable,
                    f'{self.curr_stage}/prop_visited_xzr': len(self.visited_xzr) / (int(360 / args.DT) * n_reachable),
                    f'{self.curr_stage}/num_obj_seen': n_obj_seen,
                    f'{self.curr_stage}/prop_obj_seen': n_obj_seen / self.total_pickupable_or_openable_objects,
                }
            }

        return self.metrics

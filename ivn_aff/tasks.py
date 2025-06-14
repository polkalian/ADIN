import gym,random, torch, math, skimage, time
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Union, Sequence, cast

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.system import get_logger

from ivn_aff.environment import IThorEnvironment
from ivn_aff.constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_DOWN,
    LOOK_UP,
    END,
    DIRECTIONAL_AHEAD_PUSH,
    DIRECTIONAL_BACK_PUSH,
    DIRECTIONAL_RIGHT_PUSH,
    DIRECTIONAL_LEFT_PUSH,
    PICK_UP,
    DROP
)
import torch.nn as nn

from allenact_plugins.ithor_plugin.ithor_util import (
    round_to_factor,
    include_object_data,
)
import ivn_aff.utils.pose as pu
from ivn_aff.utils.fmm_planner import FMMPlanner
import skimage.morphology as sm
import skimage

# import warnings
# warnings.filterwarnings("ignore")

movie_dir = './movie'
num_local_steps = 10
collision_threshold = 0.20
turn_angle = 90
map_resolution = 0.05
map_size = 20
OBSTACLE_LIST = ['ArmChair', 'DogBed', 'Box', 'Chair', 'Desk', 'DiningTable', 'SideTable', 'Sofa', 'Stool', 'Television', 'Pillow', 'Bread', 'Apple', 'AlarmClock', 'Lettuce', 'GarbageCan', 'Laptop', 'Microwave', 'Pot', 'Tomato']

class ObstaclesNavTask(Task[IThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP,
                DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH, DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH,
                PICK_UP, DROP,
                END)

    def __init__(
            self,
            env: IThorEnvironment,
            sensors: List[Sensor],
            task_info: Dict[str, Any],
            max_steps: int,
            reward_configs: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_geodesic_distance = self.env.distance_to_target(self.task_info["target"])
        self.last_geodesic_distance_ = self.env.distance_to_point(self.task_info['scene'], self.task_info["target"])
        self.last_tget_in_path = False

        self.optimal_distance = self.last_geodesic_distance_
        self.optimal_distance_ = self.last_geodesic_distance
        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List[Any] = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()
        self.num_moves_made = 0

        self.push_success = 0
        self.pick_success = 0
        self.distance_change: float = 0.0
        self.push_moves = 0
        self.pick_moves = 0
        self.effective_moves = 0
        self.push_val = False
        self.pick_val = False
        self.action_str = None
        self.action = None
        self.last_action_success = False
        self.inter_reward = 0
        self.nav_reward = 0
        self.inter_moves=0
        self.episode_length = 0

        self.greedy_expert = None

        self.mode = 'nav'
        self.start = True
        # print("initialize")

        self.map_shape = (int(map_size / map_resolution),
                     int(map_size / map_resolution))
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.visited_vis = np.zeros(self.map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [map_size / 2.0,
                         map_size / 2.0, 0.]
        self.last_action = None
        self.selem = skimage.morphology.disk(2)
        self.goal_map = None
        self.goal_map = None
        self.global_goal = [200,200]
        self.count = 0
        self.goal_reached = False
        self.goal_location = 0
        self.inter_mode = 0
        self.pick_index = 0
        self.inter_index = 0
        self.goal_reward = False
        self.subgoal_reached = 0
        self.target_out_of_map = False
        self.found_target = False
        self.tmp = np.zeros((2))
        self.anti_move = np.zeros(self.map_shape).astype('bool')
        self.anti_pick = np.zeros(self.map_shape).astype('bool')
        self.push_dir = 'random'
        self.cpu_action = [0.5,0.5]
        self.d_goal = [0,0]

        self.reset_map()
    

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_x, start_y
        start = [int(r - gx1),
                 int(c - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        traversible = skimage.morphology.binary_erosion(
            map_pred[gx1:gx2, gy1:gy2],
            self.selem)
        # traversible = map_pred[gx1:gx2, gy1:gy2]
        traversible[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2] == 1] = 1
        traversible[int(start[0] - gx1) - 4:int(start[0] - gx1) + 5,
                    int(start[1] - gy1) - 4:int(start[1] - gy1) + 5] = 1

        # 若目标不可达，转化为可达区域上的10个点
        goal = np.copy(planner_inputs['goal'])

        # 判断是否到达目标点
        tmp = np.where(goal == 1)
        dist = [[np.abs(tmp[0][i]-start_x), np.abs(tmp[1][i]-start_y)] for i in range(tmp[0].shape[0])]
        for d in dist:    
            if d[0] < 4.95 and d[1] < 4.95:
                self.goal_reached = True

        # Collision check
        if self.last_action == 0:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 1 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 1 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wx, wy
                        r, c = int(r), \
                            int(c)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, stop = self._get_stg(traversible, start, np.copy(goal),
                                  planning_window)

        # Deterministic Local Policy
        if self.goal_reached and planner_inputs['found_goal'] == 1:
            action = 11  # Stop
        else:
            # if stop:
            #     self.goal_reached = True
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            # print([stg_x, stg_y], start)
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > turn_angle / 2.:
                action = 1  # Right
            elif relative_angle < -turn_angle / 2.:
                action = 2  # Left
            else:
                action = 0  # Forward

        return action
    

    def _get_stg(self, traversible, start, goal, planning_window):
        """Get short-term goal"""

        # [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        # x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop


    def act(self, action_str):
        # action_str = MOVE_AHEAD
        if action_str == END:
            self.mode='nav'
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        elif action_str == PICK_UP:
            if self.mode == 'nav':
                self.mode = 'pick'
            self.pick_moves += 1
            obj = self.env.pickupable_closest_obj_by_types(self.task_info["obstacles_types"])
            self.pick_val = obj is not None
            if obj != None:
                self.env.step({"action": action_str,
                            "objectId": obj["objectId"],
                            })
                self.env.step({"action": 'HideObject',
                            "objectId": obj["objectId"],
                            })
                self.last_action_success = self.env.last_action_success
            else:
                self.last_action_success = False
            if self.last_action_success:
                self.pick_success += 1
                # print("pick successful")

        elif action_str == DROP:
            if self.mode == 'nav':
                self.mode = 'pick'
            self.pick_moves += 1
            obj = self.env.last_event.metadata['inventoryObjects']
            if len(obj) > 0:
                obj = obj[0]
                self.env.step({"action": 'UnhideObject',
                                "objectId": obj['objectId'],
                                })
                self.env.controller.step(
                    {
                    "action": "ThrowObject",
                    "moveMagnitude": 20,
                    "forceAction": True
                    }
                )
                self.last_action_success = self.env.last_action_success
                if self.last_action_success:
                    self.pick_success += 1
                    # print('drop successful')
        elif action_str in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH,
                            DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
            if self.mode == 'nav':
                self.mode = 'move'
            angle = [0.001, 180, 90, 270][self.action - 5]
            obj = self.env.moveable_closest_obj_by_types(self.task_info["obstacles_types"])
            self.push_val = obj is not None
            if obj != None:
                self.env.step({"action": action_str,
                            "objectId": obj["objectId"],
                            "moveMagnitude": obj["mass"] * 200,
                            "pushAngle": angle})
                self.env.initialize(0.25, renderObjectImage=True, renderDepthImage=True)
                self.last_action_success = self.env.last_action_success
            else:
                self.last_action_success = False

            # 决策层训练
            self.push_moves += 1
            if self.last_action_success:
                self.push_success += 1

        elif action_str in [LOOK_UP, LOOK_DOWN]:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        if len(self.path) > 1 and self.path[-1] != self.path[-2]:
            self.num_moves_made += 1

    
    def contour_centers(self, map):
        map = sm.binary_opening(map, sm.disk(1))
        map = sm.binary_closing(map, sm.disk(1))

        contours = skimage.measure.find_contours(map, 0.1)
        centers = [np.array([np.mean(c[:,0]),np.mean(c[:,1])]) for c in contours]
        return centers, contours
        
    
    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action) -> RLStepResult:

        self.inter_mode = 0

        self.cpu_action = action
        # print(cpu_action)

        observation = self.get_observations(inter_mode=self.inter_mode)
        explored_map = observation['maps']['explored'].astype('bool')
        traversible_map = observation['maps']['traversible'].astype('bool')
        target_map = observation['maps']['semantic'][:,:,3].astype('bool')
        semantic_map = observation['maps']['semantic']
        moveable_map = observation['maps']['semantic'][:,:,5].astype('bool').T
        pickable_map = observation['maps']['semantic'][:,:,6].astype('bool').T
        planner_pose_inputs = observation['maps']['pose_pred']
        visible_map = observation['maps']['semantic'][:,:,7].astype('bool').T
        local_map = observation['maps']['semantic'][:,:,4].astype('bool').T
        rgb = self.env.current_frame.copy()

        # 生成long-term goal对应的map
        self.global_goal = [int(self.cpu_action[0] * 100-50+planner_pose_inputs[0]), int(self.cpu_action[1] * 100-50+planner_pose_inputs[1])]
        self.global_goal = [min(self.global_goal[0], int(self.map_shape[0] - 1)), min(self.global_goal[1], int(self.map_shape[1] - 1))]
        self.global_goal = [max(self.global_goal[0], 0), max(self.global_goal[1], 0)]
        self.goal_map = np.zeros((self.map_shape[0], self.map_shape[1]))
        try:
            self.goal_map[self.global_goal[0], self.global_goal[1]] = 1
        except:
            print('goal_map exception')
        self.goal_map = self.goal_map.astype('bool')
        self.d_goal = np.sqrt(np.sum(np.square(self.global_goal - np.array([np.where(target_map==1)[0][0],np.where(target_map==1)[1][0]]))))
        # print(self.global_goal, planner_pose_inputs[:2], self.cpu_action)

        visible_moveable_map = moveable_map & ~self.anti_move  #  & visible_map
        visible_pickable_map = pickable_map & ~self.anti_pick  # & visible_map
        visible_moveable_map = skimage.morphology.binary_dilation(visible_moveable_map, skimage.morphology.disk(10))
        visible_pickable_map = skimage.morphology.binary_dilation(visible_pickable_map, skimage.morphology.disk(10))
        # self.goal_map = target_map

        # 目标点是否在局部图外，是则先等价到最近traversible点上
        self.goal_map_ = self.goal_map
        if not True in (explored_map & self.goal_map):
            r_p = np.where((explored_map & traversible_map) == 1)
            r_p = [np.array([r_p[0][i], r_p[1][i]]) for i in range(r_p[0].shape[0])]
            dist = [np.sum(np.square(c-np.array(self.global_goal))) for c in r_p]
            new_goal = r_p[dist.index(min(dist))].astype('int')
            self.goal_map_ = np.zeros((self.map_shape[0], self.map_shape[1]))
            self.goal_map_[new_goal[0], new_goal[1]] = 1
        
        # 如果目标点点可达
        reachable_map = (explored_map & traversible_map) | self.goal_map_.astype('bool')
        reachable_map = skimage.morphology.binary_erosion(reachable_map, self.selem)
        labelled_map = skimage.measure.label(reachable_map, connectivity=1)
        found_target = labelled_map[np.where(self.goal_map_ == 1)[0][0],np.where(self.goal_map_ == 1)[1][0]] == labelled_map[int(planner_pose_inputs[0]),int(planner_pose_inputs[1])] != 0
        if found_target:
            pass

        if visible_pickable_map[int(planner_pose_inputs[0]),int(planner_pose_inputs[1])]:
            labelled_pickable = skimage.measure.label(visible_pickable_map, connectivity=1)
            labelled_pickable = ~(labelled_pickable - labelled_pickable[int(planner_pose_inputs[0]),int(planner_pose_inputs[1])]).astype('bool')
            pickable_centers, _ = self.contour_centers(labelled_pickable)
            angle = np.rad2deg(np.arctan2(pickable_centers[0][0]-planner_pose_inputs[0], pickable_centers[0][1]-planner_pose_inputs[1])) % 360.0

            # 开始交互时面向物体
            d_angles = [np.abs(angle - rotation) if np.abs(angle - rotation) <= 180.0 else 360.0 - np.abs(angle - rotation) for rotation in [0.0, 90.0, 180.0, 270.0]]
            rotation = 0.0 + d_angles.index(min(d_angles))*90.0
            self.env.controller.step(
                action="Teleport",
                rotation=dict(x=0, y=rotation, z=0),
            )
            if self.env.last_action_success:
                planner_pose_inputs[2] = rotation
            d_angle = np.abs(angle - planner_pose_inputs[2]) if np.abs(angle - planner_pose_inputs[2]) <= 180.0 else 360.0 - np.abs(angle - planner_pose_inputs[2])

            if d_angle < 45.0:
            # if True:
                if False in (self.tmp == planner_pose_inputs[:2]).tolist():
                    self.inter_index = 0
                else:
                    self.inter_index += 1
                if self.inter_index <= 5:
                    self.inter_mode = 2
                    self.pick_index = 0
                else:
                    self.anti_pick |= skimage.morphology.binary_erosion(labelled_pickable, skimage.morphology.disk(10))
                self.tmp = planner_pose_inputs[:2]
        if visible_moveable_map[int(planner_pose_inputs[0]),int(planner_pose_inputs[1])]:
            labelled_moveable = skimage.measure.label(visible_moveable_map, connectivity=1)
            labelled_moveable = ~(labelled_moveable - labelled_moveable[int(planner_pose_inputs[0]),int(planner_pose_inputs[1])]).astype('bool')
            moveable_centers, _ = self.contour_centers(labelled_moveable)
            angle = np.rad2deg(np.arctan2(moveable_centers[0][0]-planner_pose_inputs[0], moveable_centers[0][1]-planner_pose_inputs[1])) % 360.0

            # 开始交互时面向物体
            d_angles = [np.abs(angle - rotation) if np.abs(angle - rotation) <= 180.0 else 360.0 - np.abs(angle - rotation) for rotation in [0.0, 90.0, 180.0, 270.0]]
            rotation = 0.0 + d_angles.index(min(d_angles))*90.0
            self.env.controller.step(
                action="Teleport",
                rotation=dict(x=0, y=rotation, z=0),
            )
            if self.env.last_action_success:
                planner_pose_inputs[2] = rotation
            d_angle = np.abs(angle - planner_pose_inputs[2]) if np.abs(angle - planner_pose_inputs[2]) <= 180.0 else 360.0 - np.abs(angle - planner_pose_inputs[2])

            if d_angle < 45.0:
            # if True:
                if False in (self.tmp == planner_pose_inputs[:2]).tolist():
                    self.inter_index = 0
                else:
                    self.inter_index += 1
                if self.inter_index <= 5:
                    if self.inter_mode == 2:
                        self.inter_mode = random.choice([1,1,2])
                    else:
                        self.inter_mode = 1
                else:
                    self.anti_move |= skimage.morphology.binary_erosion(labelled_moveable, skimage.morphology.disk(10))
                self.tmp = planner_pose_inputs[:2]

                self.push_dir = 'right'
                if self.inter_index == 1:
                    self.push_dir = 'forward'
                elif self.inter_index == 2:
                    self.push_dir = 'left'
                elif self.inter_index > 2:
                    self.push_dir = 'random'
        
        # rgb = observation['rgb']

        self.goal_reached = False
        k = True
        # self.goal_reward = 0

        # 开始循环
        for l_step in range(num_local_steps):
            self.episode_length += 1
            if l_step > 0:
                observation = self.get_observations(inter_mode=self.inter_mode)
                explored_map = observation['maps']['explored'].astype('bool')
                traversible_map = observation['maps']['traversible'].astype('bool')
                target_map = observation['maps']['semantic'][:,:,3].astype('bool')
                semantic_map = observation['maps']['semantic']
                moveable_map = observation['maps']['semantic'][:,:,5].astype('bool').T
                pickable_map = observation['maps']['semantic'][:,:,6].astype('bool').T
                planner_pose_inputs = observation['maps']['pose_pred']
                # rgb = observation['rgb'][...,::-1]
                rgb = self.env.current_frame.copy()

            # 目标在地图外
            if 0 in np.where(target_map == 1)[0] or self.map_shape[0]-1 in np.where(target_map == 1)[0] or 0 in np.where(target_map == 1)[1] or self.map_shape[1]-1 in np.where(target_map == 1)[1]:
                self.target_out_of_map = True

            # test
            # region
            # if np.mean(moveable_map) != 0 and k:
            #     tmp = np.where(moveable_map.T==1)
            #     index = random.choice([i for i in range(tmp[0].shape[0])])
            #     self.goal_map = np.zeros((self.map_shape[0], self.map_shape[1]))
            #     self.goal_map[tmp[0][index], tmp[1][index]] = 1
            #     self.global_goal = [tmp[0][index], tmp[1][index]]
            #     k = False
            # elif np.mean(pickable_map) != 0 and k:
            #     tmp = np.where(pickable_map.T==1)
            #     index = random.choice([i for i in range(tmp[0].shape[0])])
            #     self.goal_map = np.zeros((self.map_shape[0], self.map_shape[1]))
            #     self.goal_map[tmp[0][index], tmp[1][index]] = 1
            #     self.global_goal = [tmp[0][index], tmp[1][index]]
            #     k = False
            
            # 判断目标点位置
            # if True in (explored_map.T & traversible_map.T & self.goal_map.astype('bool')):
            #     self.goal_location = 0 # 落在可达区域
            #     # if l_step == 0:
            #     #     self.goal_reward = 0.01
            # elif True in (moveable_map.T & self.goal_map.astype('bool')):
            #     self.goal_location = 1 # 落在可推动障碍物上
            #     dist = [np.sum(np.square(c-np.array(self.global_goal))) for c in moveable_centers]
            #     new_goal = moveable_centers[dist.index(min(dist))].astype('int')
            #     # self.goal_map = np.zeros((self.map_shape[0], self.map_shape[1]))
            #     # self.goal_map[new_goal[0], new_goal[1]] = 1
            # elif True in (pickable_map.T & self.goal_map.astype('bool')):
            #     self.goal_location = 2 # 落在可拾起障碍物上
            #     dist = [np.sum(np.square(c-np.array(self.global_goal))) for c in pickable_centers]
            #     new_goal = pickable_centers[dist.index(min(dist))].astype('int')
            #     # self.goal_map = np.zeros((self.map_shape[0], self.map_shape[1]))
            #     # self.goal_map[new_goal[0], new_goal[1]] = 1
            # else:
            #     self.goal_location = 3 # 落在需要等价的区域，包括未探索和不可交互物上
            #     r_p = np.where((explored_map.T & traversible_map.T) == 1)
            #     r_p = [np.array([r_p[0][i], r_p[1][i]]) for i in range(r_p[0].shape[0])]
            #     r_p_ = r_p + moveable_centers + pickable_centers
            #     dist = [np.sum(np.square(c-np.array(self.global_goal))) for c in r_p_]
            #     new_goal = r_p_[dist.index(min(dist))].astype('int')
            #     # self.goal_map = np.zeros((self.map_shape[0], self.map_shape[1]))
            #     # self.goal_map[new_goal[0], new_goal[1]] = 1
            #     # 如果等价到障碍物上，修改位置标示
            #     if len(r_p+moveable_centers) > dist.index(min(dist)) >= len(r_p):
            #         self.goal_location = 1
            #     elif len(r_p_) > dist.index(min(dist)) >= len(r_p+moveable_centers):
            #         self.goal_location = 2
            #     else:
            #         self.goal_location = 0       
            # endregion     

            # 如果终点可达
            reachable_map = explored_map & traversible_map | target_map
            labelled_map = skimage.measure.label(reachable_map, connectivity=1)
            self.found_target = labelled_map[np.where(target_map == 1)[0][0],np.where(target_map == 1)[1][0]] == labelled_map[int(planner_pose_inputs[0]),int(planner_pose_inputs[1])] != 0
            if self.found_target:
                self.goal_map = target_map
                if not self.goal_reward:
                    self.goal_reward = True

            if self.inter_mode != 0:
                # 进入交互阶段
                if self.inter_mode == 1:
                    if self.push_dir == 'random':
                        self.action = random.choice([5,7,8])
                    elif self.push_dir == 'right':
                        self.action = 7
                    elif self.push_dir == 'left':
                        self.action = 8
                    elif self.push_dir == 'forward':
                        self.action = 5
                    action_str = self.action_names()[self.action]
                    if l_step >= 5:
                        break
                elif self.inter_mode == 2:
                    pick_sequence = [PICK_UP, ROTATE_RIGHT, ROTATE_RIGHT, DROP, ROTATE_RIGHT, ROTATE_RIGHT, 'done']
                    action_str = pick_sequence[self.pick_index]
                    self.pick_index += 1
                    if self.pick_index > 6:
                        break
            else:
                # 进入导航阶段
                # 如果到达目标点，寻找下一个目标，或进入交互阶段
                if self.goal_reached:
                    self.subgoal_reached += 1
                    break

                p_input = {}
                if self.count < 10:
                    traversible_map = traversible_map | skimage.morphology.binary_dilation(moveable_map, skimage.morphology.disk(3)) |skimage.morphology.binary_dilation(pickable_map, skimage.morphology.disk(3))
                p_input['map_pred'] = traversible_map.astype('float')
                p_input['exp_pred'] = semantic_map[:,:,1].T
                p_input['pose_pred'] = planner_pose_inputs
                p_input['goal'] = self.goal_map
                p_input['new_goal'] = l_step == num_local_steps - 1
                p_input['found_goal'] = self.found_target
                p_input['goal_reachable'] = self.goal_location == 0

                action_id = self._plan(p_input)

                action_str = self.action_names()[action_id]

                # 如果长时间卡住怎么办
                if len(self.path) > 2:
                    if self.path[-1] == self.path[-2]:
                        self.count += 1
                    elif len(set([(p['x'],p['z']) for p in self.path[-5:]])) >= 3 and len(self.path) >= 5:
                        self.count = 0
                if self.count >= 5:
                    action_str = random.choice([MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, MOVE_AHEAD])

            if self.episode_length >= 500:
                action_str = END
            self.action_str = action_str
            if self.inter_mode != 1:
                self.action = self.action_names().index(self.action_str)
            self.last_action = self.action

            #执行动作
            self.act(action_str)

            if action_str in [END]:
                break

        # 如果推动，转一圈刷新地图
        if self.inter_mode == 1:
            for i in range(4):
                self.act(ROTATE_RIGHT)
                if i in [0,2]:
                    self.get_observations(inter_mode=self.inter_mode)
                rgb = self.env.current_frame.copy()

        self.inter_mode = 0 # 序列交互后已刷新地图，无需再更新一次
        step_result = RLStepResult(
            observation=self.get_observations(inter_mode=self.inter_mode),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        self.start = False

        return step_result

    def extra_terminal(self) -> bool:
        # nav
        goal_in_range = self._is_goal_in_range()
        if isinstance(goal_in_range, bool):
            if goal_in_range:
                self._success = True
            return goal_in_range
        else:
            return False

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        tget = self.task_info["target"]
        # dist = self.dist_to_target()
        dist = self.env.distance_to_point(self.task_info['scene'], self.task_info["target"])

        if -0.5 < dist <= 0.5:
            return True
        elif dist > 0.5:
            return False
        else:
            get_logger().debug(
                "No path for {} from {} to {}".format(
                    self.env.scene_name, self.env.agent_state(), tget
                )
            )
            # print("none")
            return False

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.last_geodesic_distance
        if self.action_str != MOVE_AHEAD:
            if (
                    self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5
            ):  # (robothor limits)
                rew += 0

        if self.last_geodesic_distance == -1.0 and geodesic_distance > 0:
            rew = 1.0
        self.last_geodesic_distance = geodesic_distance

        if self.action_str in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH, DIRECTIONAL_RIGHT_PUSH,
                               DIRECTIONAL_LEFT_PUSH, PICK_UP, DROP]:
            self.inter_reward += rew
        else:
            self.nav_reward += rew

        return rew * self.reward_configs["shaping_weight"]

    def shaping_by_path(self) -> float:
        reward = 0.0
        geodesic_distance_ = self.dist_to_target_()
        if geodesic_distance_ == -1.0:
            geodesic_distance_ = self.last_geodesic_distance_
        if self.action_str in [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END]:
            if (
                    self.last_geodesic_distance_ > -0.5 and geodesic_distance_ > -0.5
            ):  # (robothor limits)
                reward += self.last_geodesic_distance_ - geodesic_distance_
                self.nav_reward += reward
        self.last_geodesic_distance_ = geodesic_distance_
        return reward

    def shaping_by_action(self) -> float:
        reward = 0.0

        if self._is_goal_in_range() and self.action_str is not END:
            self.effective_moves = 1
        return reward

    def judge(self) -> float:
        """Judge the last event."""

        if not self.found_target:
            reward = self.shaping_by_path()     #导航
        else:
            reward = 0
        reward += self.shaping()  #交互

        if self._took_end_action:
            if self._success is not None:
                # dist2tget = self.dist_to_target()
                reward += (
                    self.reward_configs["goal_success_reward"]/4
                    # 0
                    if self.goal_reward # self._success
                    else 0
                )

        self._rewards.append(float(reward))
        return float(reward)

    def spl(self):
        if not self._success:
            return 0.0
        li = self.optimal_distance
        pi = self.num_moves_made * self.env._grid_size
        res = li / (max(pi, li) + 1e-8)
        return res

    def sel(self):
        if not self._success:
            return 0.0
        res = (self.optimal_distance/self.env._grid_size)/self.episode_length
        return res

    def dist_to_target(self):
        # return self.env.distance_to_point(self.env.scene_name, self.task_info["target"])
        return self.env.distance_to_target(self.task_info["target"])

    def dist_to_target_(self):
        return self.env.distance_to_point(self.task_info['scene'], self.task_info["target"])
        # return self.env.distance_to_target(self.task_info["target"])

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        total_reward = float(np.sum(self._rewards))
        self._rewards = []

        # if self._success is None:
        #     return {}

        dist2tget = self.dist_to_target_()
        if dist2tget == -1:
            dist2tget = self.last_geodesic_distance
        spl = self.spl()
        sel = self.sel()
        return {
            **super(ObstaclesNavTask, self).metrics(),
            "success": self._success,  # False also if no path to target
            "total_reward": total_reward,
            "dist_to_target": dist2tget,
            "num_moves_made": self.num_moves_made,
            # "nav_moves": self.nav_moves,
            # "nav_success": self.nav_success,
            "spl": spl,
            'sel': sel,
            # "target_in_reachable_points": self.last_tget_in_path,
            "push_success": self.push_success,
            "push_moves": self.push_moves,
            "pick_success": self.pick_success,
            "pick_moves": self.pick_moves,
            # "distance_change": self.distance_change,
            # "effective_moves": self.effective_moves,
            "inter_reward": self.inter_reward,
            "nav_reward": self.nav_reward,
            "blocked_ratio": self.optimal_distance_ == -1,
            "blocked_final": self.dist_to_target() == -1,
            "goal_reward": self.goal_reward,
            "subgoal_reached": self.subgoal_reached,
            "target_out_of_map": self.target_out_of_map,
            "cpu_x": self.cpu_action[0],
            "cpu_y": self.cpu_action[1],
            "d_gaol": self.d_goal
        }


class ObjectPlacementTask(Task[IThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP,
                DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH, DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH,
                END)

    def __init__(
            self,
            env: IThorEnvironment,
            sensors: List[Sensor],
            task_info: Dict[str, Any],
            max_steps: int,
            reward_configs: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_geodesic_distance = self.env.distance_to_point(
            self.env.scene_name, self.task_info["target"]
        )
        self.obj_last_geodesic_distance = self.obj_dist_to_target()
        self.last_both_in_path = False

        self.optimal_distance = self.last_geodesic_distance + self.obj_last_geodesic_distance
        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List[Any] = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()
        self.num_moves_made = 0

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:

        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.action_names()[action]

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        elif action_str in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH,
                            DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
            angle = [0.001, 180, 90, 270][action - 5]
            obj = self.env.moveable_closest_obj_by_types(self.task_info["obstacles_types"])
            if obj != None:
                self.env.step({"action": action_str,
                               "objectId": obj["objectId"],
                               "moveMagnitude": obj["mass"] * 100,
                               "pushAngle": angle,
                               "autoSimulation": False})
                self.last_action_success = self.env.last_action_success
            else:
                self.last_action_success = False
        elif action_str in [LOOK_UP, LOOK_DOWN]:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        if len(self.path) > 1 and self.path[-1] != self.path[-2]:
            self.num_moves_made += 1
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )

        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        objs = self.env.get_objects_by_type(self.task_info["object_type"])
        tgt_obj = self.env.get_objects_by_type(self.task_info["target_type"])[0]
        for obj in objs:
            if obj["objectId"] in tgt_obj["receptacleObjectIds"]:
                return True

        tget = self.task_info["target"]
        dist = self.obj_dist_to_target()

        if -0.5 < dist <= 0.2:
            return True
        elif dist > 0.2:
            return False
        else:
            get_logger().debug(
                "No path for {} from {} to {}".format(
                    self.env.scene_name, self.env.agent_state(), tget
                )
            )
            return None

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.obj_dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.obj_last_geodesic_distance
        if (
                self.obj_last_geodesic_distance > -0.5 and geodesic_distance > -0.5
        ):  # (robothor limits)
            rew += self.obj_last_geodesic_distance - geodesic_distance
        self.obj_last_geodesic_distance = geodesic_distance

        return rew * self.reward_configs["shaping_weight"]

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()

        if self._took_end_action:
            if self._success is not None:
                reward += (
                    self.reward_configs["goal_success_reward"]
                    if self._success
                    else self.reward_configs["failed_stop_reward"]
                )

        self._rewards.append(float(reward))
        return float(reward)

    def spl(self):
        if not self._success:
            return 0.0
        li = self.optimal_distance
        pi = self.num_moves_made * self.env._grid_size
        res = li / (max(pi, li) + 1e-8)
        return res

    def dist_to_target(self):
        objs, idx = self.env.get_objects_and_idx_by_type(self.task_info["object_type"])
        dis = []
        for id in idx:
            dis.append(self.env.object_distance_to_point(self.env.scene_name, id, self.task_info["target"]))
        id = idx[np.argmin(dis)]
        return self.env.distance_to_point(self.env.scene_name, self.env.all_objects()[id]["position"])

    def obj_dist_to_target(self):
        objs, idx = self.env.get_objects_and_idx_by_type(self.task_info["object_type"])
        dis = []
        for id in idx:
            dis.append(self.env.object_distance_to_point(self.env.scene_name, id, self.task_info["target"]))
        return min(dis)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        total_reward = float(np.sum(self._rewards))
        self._rewards = []

        if self._success is None:
            return {}

        dist2tget = self.obj_dist_to_target()
        spl = self.spl()

        return {
            **super(ObjectPlacementTask, self).metrics(),
            "success": self._success,  # False also if no path to target
            "total_reward": total_reward,
            "dist_to_target": dist2tget,
            "spl": spl,
            "both_in_reachable_points": self.last_both_in_path,
        }

    def query_expert(self, end_action_only: bool = False, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True
        if end_action_only:
            return 0, False
        else:
            raise NotImplementedError
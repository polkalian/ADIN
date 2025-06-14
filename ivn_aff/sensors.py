from typing import Any, Dict, Optional, List, Tuple, Union

import gym, quaternion, torch
import numpy as np
import abc

from ivn_aff.environment import IThorEnvironment
from ivn_aff.tasks import ObjectPlacementTask, ObstaclesNavTask
from allenact.embodiedai.sensors.vision_sensors import Sensor, RGBSensor, DepthSensor
from allenact.base_abstractions.sensor import AbstractExpertSensor
from allenact.base_abstractions.task import Task, SubTaskType
from allenact.utils.misc_utils import prepare_locals_for_super
from ivn_aff.utils.utils_3d_torch import (
    get_corners,
    local_project_2d_points_to_3d,
    project_2d_points_to_3d,
)
from allenact.utils.experiment_utils import (
    LoggingPackage,
    ScalarMeanTracker,
    set_deterministic_cudnn,
    set_seed,
)

from ivn_aff.mapper import Mapper
from ivn_aff.utils.utils import get_camera_matrix
# from ivn_aff.utils.segmentation import SegmentationHelper
import skimage
from PIL import Image
from ivn_aff.visualization import Animation, visualize_segmentationRGB, visualize_topdownSemanticMap
from ivn_aff.yolov7.yolov7_seg import YOLOV7_SegmentationHelper
import torchvision.models 
import torchvision.transforms as transforms
import torchvision.ops as ops
import torch.nn as nn
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

fov = 90.0
map_size = 20
map_resolution = 0.05
STEP_SIZE = 0.25
OBSTACLE_LIST = ['ArmChair', 'DogBed', 'Box', 'Chair', 'Desk', 'DiningTable', 'SideTable', 'Sofa', 'Stool', 'Television', 'Pillow', 'Bread', 'Apple', 'AlarmClock', 'Lettuce', 'GarbageCan', 'Laptop', 'Microwave', 'Pot', 'Tomato']
AFFORDANCE_LIST = ['moveable', 'pickupable', 'visible']
# OBSTACLE_LIST = ['ArmChair', 'DogBed']
OBSTACLE_to_ID = {
    OBSTACLE_LIST[i]: i+len(AFFORDANCE_LIST) for i in range(len(OBSTACLE_LIST))
}
INSTANCE_SEG_THRESHOLD = 0.8

create_movie = False
import  time

class MapsSensorThor(Sensor):
    """Sensor for Maps in iTHOR.

    Returns from a running IThorEnvironment instance, the current
    top-down maps corresponding to the agent's egocentric view.
    """
    def __init__(self,
                 seg_utils,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 uuid="maps"):
        self.seg_utils = seg_utils
        observation_space = (
            gym.spaces.Dict({
            'semantic': gym.spaces.Box(low=0.0,
                                    high=1.0,
                                    shape=(int(map_size/map_resolution), int(map_size/map_resolution), len(OBSTACLE_LIST)+len(AFFORDANCE_LIST)),
                                    dtype=np.float32),
            'traversible': gym.spaces.MultiBinary([int(map_size/map_resolution),int(map_size/map_resolution)]),
            'explored': gym.spaces.MultiBinary([int(map_size/map_resolution),int(map_size/map_resolution)]),
                             }))
        # initialize mapper
        self.w = width
        self.h = height
        self.reset()
        # print(observation_space['seg'].shape)

        if self.seg_utils == "yolov7":
            self.device = torch.device('cuda:6')
            device_cpu = torch.device('cpu')
            # from copy import deepcopy
            with torch.no_grad():
                # self.segmentation_helper = YOLOV7_SegmentationHelper(self.device)
                # self.segmentation_helper = deepcopy(self.segmentation_helper)

                # resnet18用于提取特征
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context

                # self.feature_extractor = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT).to(device_cpu)
                self.feature_extractor = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).to(device_cpu)
                # self.feature_extractor = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.DEFAULT).to(device_cpu)
                self.feature_extractor=nn.Sequential(*list(self.feature_extractor.children())[:4])
                self.feature_extractor.eval()
                self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

        super().__init__(**prepare_locals_for_super(locals()))


    def reset(self):
        C = get_camera_matrix(self.w, self.h, fov=fov)
        self.position = {'x': 0,
                         'y': 1.5760,  # fixed when standing up
                         'z': 0}
        self.start_position = self.position
        self.rotation = 0.0
        self.start_rotation = self.rotation
        self.map_size = map_size
        self.resolution = map_resolution
        self.max_depth = 10
        self.z = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
                  1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2]
        self.STEP_SIZE = STEP_SIZE
        loc_on_map_size = int(
            np.floor(self.STEP_SIZE / self.resolution / 2))  # +5
        self.loc_on_map_selem = np.ones(
            (loc_on_map_size * 2 + 1, loc_on_map_size * 2 + 1)).astype(bool)
        # self.mapper = Mapper(C, self.position, self.map_size, self.resolution,
        #                      max_depth=self.max_depth, z_bins=self.z, num_categories=10, 
        #                      loc_on_map_selem=self.loc_on_map_selem) # len(OBSTACLE_LIST)+len(AFFORDANCE_LIST)

        # self.selem = skimage.morphology.disk(
        #     int(3 * (0.05 / self.resolution)))  # resolution是0.02的时候这里是 2 * 0.02/self.resolution
        # self.selem_agent_radius = skimage.morphology.disk(
        #     int(3 * (0.05 / self.resolution)))  # although the agent's radius is 0.2m  #resolution是0.02的时候这里是0.06

        self.start = True
        self.inter_mode = False
        self.pick_count = 0

        if create_movie:
            self.vis = Animation(self.w, self.h)
        else:
            self.vis = None


    def get_observation(
            self,
            env: IThorEnvironment,
            task: ObstaclesNavTask,
            *args: Any,
            **kwargs: Any
    ) -> Any:
        # update the map
        rgb_prev = env.current_frame.copy()
        depth_prev = env.current_depth.copy()
        pos_prev = env.agent_state().copy()

        if 'inter_mode' in kwargs.keys():
            self.inter_mode = False
            if (kwargs['inter_mode'] == 2 and self.pick_count in [0,3]) or kwargs['inter_mode'] == 1:
                self.inter_mode = True
                # print('interaction mode!')
            if kwargs['inter_mode'] == 2:
                self.pick_count += 1

        # 分割过程------------------------------------------------------------------------------
        if self.seg_utils == 'GT':
            instance_seg = np.zeros((self.h, self.w, 10))
            segmented_dict = {
                'scores': [],
                'categories': [],
                'masks': [],
            }
            all_features = 0
            
            instance_masks = env.current_instance_masks
            all_obj = env.all_objects()
            for objectId, mask in instance_masks.items():
                category = objectId.split('|')[0]
                if '_' in category:
                    category = category.split('_')[0]
                if category in OBSTACLE_LIST:
                    category_id = OBSTACLE_to_ID[category]
                    if category_id < 10:
                        instance_seg[:, :, category_id] += mask.astype('float')

                    segmented_dict['scores'].append(1.0)
                    segmented_dict['categories'].append(category)
                    segmented_dict['masks'].append(mask.astype('bool'))
                    for o in all_obj:
                        if o['objectId'] == objectId:
                            if o['moveable']:
                                instance_seg[:,:,0] += mask.astype('float')
                            if o['pickupable']:
                                instance_seg[:,:,1] += mask.astype('float')
                            if o['visible']:
                                instance_seg[:,:,2] += mask.astype('float')
            seg_prev = instance_seg.astype('bool').astype('float')
        
        # elif self.seg_utils == 'yolov7':
        #     instance_seg = np.zeros((self.h, self.w, 10))
        #     segmented_dict = []
        #     self.segmentation_helper.model.to(self.device)
        #     self.feature_extractor.to(self.device)

        #     segmented_list = self.segmentation_helper.seg_pred(rgb_prev)
        #     rgb = Image.fromarray(rgb_prev)
        #     height, width = rgb.size
        #     img_tensor = self.transform(rgb).unsqueeze(0)
        #     features = self.feature_extractor(img_tensor.to(self.device))

        #     for obj in segmented_list:
        #         if obj['score'] < INSTANCE_SEG_THRESHOLD:
        #             continue
        #         category = obj["label"]
        #         if category in OBSTACLE_LIST:
        #             category_id = OBSTACLE_to_ID[category]
        #             mask = obj['mask']
        #             # mask = mask.astype(np.uint8)
        #             # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        #             # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, borderType = cv2.BORDER_CONSTANT,borderValue= 0)
        #             y_list, x_list = np.where(mask)
        #             if len(x_list) == 0 or len(y_list) == 0:
        #                 # print(category)
        #                 continue
        #             bbox = torch.tensor([[min(x_list), min(y_list), max(x_list), max(y_list)]])

        #             # instance_seg[:,:,category_id] += mask.astype('float')  #补充了+，考虑到同一视角下有同类别的多实例物体
        #             # if category not in segmented_dict:
        #             #     segmented_dicts[category] = {}

        #             x_min, y_min, x_max, y_max = bbox[0]
        #             if x_max == x_min:
        #                 x_max += 1
        #             if y_max == y_min:
        #                 y_max += 1

        #             normalized_bbox = torch.zeros_like(bbox, dtype=features.dtype).to(self.device)
        #             normalized_bbox[0, 0] = bbox[0, 0] / width
        #             normalized_bbox[0, 1] = bbox[0, 1] / height
        #             normalized_bbox[0, 2] = bbox[0, 2] / width
        #             normalized_bbox[0, 3] = bbox[0, 3] / height

        #             # concat_feature = visual_feature.cpu().detach().numpy()
        #             visual_feature = ops.roi_pool(features, [normalized_bbox], output_size=(7, 7)).ravel().cpu().detach().numpy()


        #             instance_info = {
        #                 'score': obj['score'],
        #                 'mask': mask.astype(bool),
        #                 'feature': visual_feature,
        #                 # 'box': normalized_bbox,
        #             }
        #             segmented_dict.append(instance_info)

        #     # 获得前十个物体的特征和全局特征拼接
        #     full_box = torch.tensor([[0,0,1,1]], dtype=features.dtype).to(self.device)
        #     all_features = [ops.roi_pool(features, [full_box], output_size=(7, 7)).ravel().cpu().detach().numpy()]
        #     segmented_dict.sort(key=lambda k: (k.get('score', 0)))
        #     if len(segmented_dict) < 10:
        #         while len(segmented_dict) < 10:
        #             segmented_dict.append({
        #                 'score': 0,
        #                 'mask': np.zeros_like(mask).astype(bool),
        #                 'feature': all_features[0],
        #             })
        #     else:
        #         segmented_dict = segmented_dict[:10]
        #     for i, seg in enumerate(segmented_dict):
        #         all_features.append(np.hstack([seg['feature'], all_features[0]]))                
        #         instance_seg[:,:,i] += seg['mask'].astype('float')
        #     all_features = np.vstack(all_features[1:])
            
        #     seg_prev = instance_seg.astype('bool').astype('float')

        # dataset collection
        elif self.seg_utils == 'yolov7':
            instance_seg = np.zeros((self.h, self.w, 10))
            segmented_dict = []
            # self.segmentation_helper.model.to(self.device)
            self.feature_extractor.to(self.device)

            # segmented_list = self.segmentation_helper.seg_pred(rgb_prev)
            instance_masks = env.current_instance_masks
            rgb = Image.fromarray(rgb_prev)
            height, width = rgb.size
            img_tensor = self.transform(rgb).unsqueeze(0)
            features = self.feature_extractor(img_tensor.to(self.device))

            all_obj = env.all_objects()
            for objectId, mask in instance_masks.items():
                category = objectId.split('|')[0]
                if '_' in category:
                    category = category.split('_')[0]
                if category in OBSTACLE_LIST:
                    y_list, x_list = np.where(mask)
                    if len(x_list) == 0 or len(y_list) == 0:
                        # print(category)
                        continue
                    bbox = torch.tensor([[min(x_list), min(y_list), max(x_list), max(y_list)]])

                    # instance_seg[:,:,category_id] += mask.astype('float')  #补充了+，考虑到同一视角下有同类别的多实例物体
                    # if category not in segmented_dict:
                    #     segmented_dicts[category] = {}

                    x_min, y_min, x_max, y_max = bbox[0]
                    if x_max == x_min:
                        x_max += 1
                    if y_max == y_min:
                        y_max += 1

                    normalized_bbox = torch.zeros_like(bbox, dtype=features.dtype).to(self.device)
                    normalized_bbox[0, 0] = bbox[0, 0] / width
                    normalized_bbox[0, 1] = bbox[0, 1] / height
                    normalized_bbox[0, 2] = bbox[0, 2] / width
                    normalized_bbox[0, 3] = bbox[0, 3] / height

                    # concat_feature = visual_feature.cpu().detach().numpy()
                    visual_feature = ops.roi_pool(features, [normalized_bbox], output_size=(7, 7)).ravel().cpu().detach().numpy()


                    instance_info = {
                        'mask': mask.astype(bool),
                        'feature': visual_feature,
                        "objId": objectId
                        # 'box': normalized_bbox,
                    }
                    segmented_dict.append(instance_info)

            # 获得前十个物体的特征和全局特征拼接
            full_box = torch.tensor([[0,0,1,1]], dtype=features.dtype).to(self.device)
            all_features = [ops.roi_pool(features, [full_box], output_size=(7, 7)).ravel().cpu().detach().numpy()]
            segmented_dict.sort(key=lambda k: (k.get('score', 0)))
            if len(segmented_dict) < 10:
                while len(segmented_dict) < 10:
                    segmented_dict.append({
                        'mask': np.zeros_like(mask).astype(bool),
                        'feature': all_features[0],
                        "objId": "none"
                    })
            else:
                segmented_dict = segmented_dict[:10]
            env.seg_ids = []
            for i, seg in enumerate(segmented_dict):
                all_features.append(np.hstack([seg['feature'], all_features[0]]))                
                instance_seg[:,:,i] += seg['mask'].astype('float')
                env.seg_ids.append(seg['objId'])
            all_features = np.vstack(all_features[1:])
            
            seg_prev = instance_seg.astype('bool').astype('float')

        # 更新地图---------------------------------------------------------------------------
        # if self.start:
        #     self.start_position = {'x': pos_prev['x'], 'y': pos_prev['y'], 'z': pos_prev['z']}
        #     self.start_rotation = pos_prev['rotation']['y']
        #     self.start = False
        # self.position = {'x': pos_prev['x'] - self.start_position['x'], 'y': self.position['y'],
        #                  'z': pos_prev['z'] - self.start_position['z']}
        # self.rotation = pos_prev['rotation']['y'] % 360.0
        # self.point_goal = [task.task_info['target']['x'] - self.start_position['x'], task.task_info['target']['z'] - self.start_position['z']]

        # self.mapper.add_observation(self.position,
        #                             self.rotation,
        #                             -30,
        #                             depth_prev,
        #                             seg_prev.astype(np.uint8),
        #                             add_obs=True,
        #                             add_seg=True,
        #                             inter_mode=self.inter_mode)

        # # 可到达点 = map - 障碍物区域
        # traversible_map = self.mapper.get_traversible_map(self.selem, 1, loc_on_map_traversible=True).T
        # # 已探索区域 = 走过点 + 已建图点
        # explored_map = self.mapper.get_explored_map(self.selem_agent_radius, point_count=1).T  # 机器人去过的地方 or map有点的地方
        # # reachable_map = traversible_map & explored_map
        # # agent 当前位置
        # position = self.mapper.get_position_on_map()
        # position_map = np.zeros((explored_map.shape[0], explored_map.shape[1]))
        # position_map[min(int(position[0]), explored_map.shape[0]-1), min(int(position[1]), explored_map.shape[1]-1)] = 1
        # pose_pred = np.zeros((7))
        # pose_pred[:2] = position
        # pose_pred[2] = self.rotation
        # pose_pred[3:] = [0, explored_map.shape[0], 0, explored_map.shape[1]]
        # # 目标点位置
        # target = self.mapper.get_target_on_map(self.point_goal)
        # target_map = np.zeros((explored_map.shape[0], explored_map.shape[1]))
        # target_map[min(int(target[0]), explored_map.shape[0]-1), min(int(target[1]), explored_map.shape[1]-1)] = 1

        # # 4. 如果有需要的话进行可视化，保存第一视角视频和地图
        # if self.vis != None:
        #     # if args.use_seg:
        #     # seg_prev_vis = visualize_segmentationRGB(rgb_prev, segmented_dict, visualize_sem_seg=True)
        #     # else:
        #     seg_prev_vis = None
        #     self.vis.add_frame(image=rgb_prev, depth=depth_prev, seg_image=seg_prev_vis, add_map=True,
        #                        add_map_seg=True, mapper=self.mapper,
        #                        point_goal=target, selem=self.selem,
        #                        selem_agent_radius=self.selem_agent_radius)

        # movie_dir = './movie'
        # # self.vis.render_movie(movie_dir, 'test')

        # semantic_map = np.sum(self.mapper.semantic_map, axis=2)
        # semantic_map = np.clip(semantic_map, 10, 100)
        # semantic_map = (semantic_map-10)/90

        # # 计算局部图
        # position = [int(position[0]),int(position[1])]
        # local_map = np.ones((explored_map.shape[0], explored_map.shape[1])).astype('bool')
        # local_map = traversible_map[position[0]-50:position[0]+50, position[1]-50:position[1]+50]
        # local_map =np.array(Image.fromarray(local_map.astype(np.uint8)).resize((400, 400))).astype('bool')

        # semantic_map = np.dstack((traversible_map.astype('float'), explored_map.astype('float'), position_map.astype('float'), target_map.astype('float'), local_map.astype('float'), semantic_map))
        # target = np.where(target_map.astype('float') == 1)
        # target = np.array([target[0][0],target[1][0]])

        return {
            # 'semantic': semantic_map,
            # 'traversible': traversible_map,
            # 'explored': explored_map,
            # 'pose_pred': pose_pred,
            # 'target': target,
            'feature': all_features,
            # 'seg_dict': segmented_dict,
        }


class RGBSensorThor(RGBSensor[IThorEnvironment, Task[IThorEnvironment]]):
    """Sensor for RGB images in iTHOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[ObjectPlacementTask]) -> np.ndarray:
        return env.current_frame.copy()


class LastRGBSensorThor(RGBSensor[IThorEnvironment, Task[IThorEnvironment]]):
    """Sensor for RGB images in iTHOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[ObjectPlacementTask]) -> np.ndarray:
        return env.last_frame.copy()


class GoalObjectTypeThorSensor(Sensor):
    def __init__(
        self,
        object_types: List[str],
        target_to_detector_map: Optional[Dict[str, str]] = None,
        detector_types: Optional[List[str]] = None,
        uuid: str = "goal_object_type_ind",
        **kwargs: Any
    ):
        self.ordered_object_types = list(object_types)
        assert self.ordered_object_types == sorted(
            self.ordered_object_types
        ), "object types input to goal object type sensor must be ordered"

        if target_to_detector_map is None:
            self.object_type_to_ind = {
                ot: i for i, ot in enumerate(self.ordered_object_types)
            }

            observation_space = gym.spaces.Discrete(len(self.ordered_object_types))
        else:
            assert (
                detector_types is not None
            ), "Missing detector_types for map {}".format(target_to_detector_map)
            self.target_to_detector = target_to_detector_map
            self.detector_types = detector_types

            detector_index = {ot: i for i, ot in enumerate(self.detector_types)}
            self.object_type_to_ind = {
                ot: detector_index[self.target_to_detector[ot]]
                for ot in self.ordered_object_types
            }

            observation_space = gym.spaces.Discrete(len(self.detector_types))

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[ObjectPlacementTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        return self.object_type_to_ind[task.task_info["object_type"]]


class GPSCompassSensorIThor(Sensor[IThorEnvironment, ObjectPlacementTask]):
    def __init__(self, uuid: str = "target_coordinates_ind", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def _compute_pointgoal(self, source_position, source_rotation, goal_position):
        direction_vector = goal_position - source_position
        direction_vector_agent = self.quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        rho, phi = GPSCompassSensorIThor.cartesian_to_polar(
            direction_vector_agent[2], -direction_vector_agent[0]
        )
        return np.array([direction_vector[0],direction_vector[2]], dtype=np.float32)

    @staticmethod
    def quaternion_from_y_angle(angle: float) -> np.quaternion:
        r"""Creates a quaternion from rotation angle around y axis
        """
        return GPSCompassSensorIThor.quaternion_from_coeff(
            np.array(
                [0.0, np.sin(np.pi * angle / 360.0), 0.0, np.cos(np.pi * angle / 360.0)]
            )
        )

    @staticmethod
    def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
        r"""Creates a quaternions from coeffs in [x, y, z, w] format
        """
        quat = np.quaternion(0, 0, 0, 0)
        quat.real = coeffs[3]
        quat.imag = coeffs[0:3]
        return quat

    @staticmethod
    def cartesian_to_polar(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    @staticmethod
    def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
        r"""Rotates a vector by a quaternion
        Args:
            quat: The quaternion to rotate by
            v: The vector to rotate
        Returns:
            np.array: The rotated vector
        """
        vq = np.quaternion(0, 0, 0, 0)
        vq.imag = v
        return (quat * vq * quat.inverse()).imag

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:

        agent_state = env.agent_state()
        agent_position = np.array([agent_state[k] for k in ["x", "y", "z"]])
        rotation_world_agent = self.quaternion_from_y_angle(
            agent_state["rotation"]["y"]
        )

        goal_position = np.array([task.task_info["target"][k] for k in ["x", "y", "z"]])

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )


class DepthSensorIThor(DepthSensor[IThorEnvironment, Task[IThorEnvironment]]):
    # For backwards compatibility
    def __init__(
            self,
            use_resnet_normalization: Optional[bool] = None,
            use_normalization: Optional[bool] = None,
            mean: Optional[np.ndarray] = np.array([[0.5]], dtype=np.float32),
            stdev: Optional[np.ndarray] = np.array([[0.25]], dtype=np.float32),
            height: Optional[int] = None,
            width: Optional[int] = None,
            uuid: str = "depth",
            output_shape: Optional[Tuple[int, ...]] = None,
            output_channels: int = 1,
            unnormalized_infimum: float = 0.0,
            unnormalized_supremum: float = 5.0,
            scale_first: bool = False,
            **kwargs: Any
    ):
        # Give priority to use_normalization, but use_resnet_normalization for backward compat. if not set
        if use_resnet_normalization is not None and use_normalization is None:
            use_normalization = use_resnet_normalization
        elif use_normalization is None:
            use_normalization = False

        super().__init__(**prepare_locals_for_super(locals()))

    def frame_from_env(self, env: IThorEnvironment, task: Optional[ObjectPlacementTask]) -> np.ndarray:
        return env.current_depth.copy()


class LastDepthSensorIThor(DepthSensor[IThorEnvironment, Task[IThorEnvironment]]):
    # For backwards compatibility
    def __init__(
            self,
            use_resnet_normalization: Optional[bool] = None,
            use_normalization: Optional[bool] = None,
            mean: Optional[np.ndarray] = np.array([[0.5]], dtype=np.float32),
            stdev: Optional[np.ndarray] = np.array([[0.25]], dtype=np.float32),
            height: Optional[int] = None,
            width: Optional[int] = None,
            uuid: str = "last_depth",
            output_shape: Optional[Tuple[int, ...]] = None,
            output_channels: int = 1,
            unnormalized_infimum: float = 0.0,
            unnormalized_supremum: float = 5.0,
            scale_first: bool = False,
            **kwargs: Any
    ):
        # Give priority to use_normalization, but use_resnet_normalization for backward compat. if not set
        if use_resnet_normalization is not None and use_normalization is None:
            use_normalization = use_resnet_normalization
        elif use_normalization is None:
            use_normalization = False

        super().__init__(**prepare_locals_for_super(locals()))

    def frame_from_env(self, env: IThorEnvironment, task: Optional[ObjectPlacementTask]) -> np.ndarray:
        return env.last_depth.copy()


class FrameSensorThor(Sensor):
    """Sensor for Class Segmentation in iTHOR.

    Returns from a running IThorEnvironment instance, the current
    class segmentation corresponding to the agent's egocentric view.
    """
    def __init__(self,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 uuid="frame"):
        observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(height, width, 3),
            dtype=np.float64,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        return env.current_frame.copy()


class ClassSegmentationSensorThor(Sensor):
    """Sensor for Class Segmentation in iTHOR.

    Returns from a running IThorEnvironment instance, the current
    class segmentation corresponding to the agent's egocentric view.
    """
    def __init__(self,
                 objectTypes,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 uuid="seg"):
        self.objectTypes = sorted(list(objectTypes))
        observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(height, width, len(objectTypes)),
            dtype=np.float64,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        if not env.using_mask_rcnn:
            return env.get_masks_by_object_types(self.objectTypes).copy()
        else:
            output = env.get_mask_rcnn_result()
            labels = list(output["labels"].detach().cpu().numpy())
            masks = output["masks"].squeeze(1).detach().cpu().numpy()
            mask = np.ones((env.current_frame.shape[0], env.current_frame.shape[0])) * len(self.objectTypes)
            for idx, mask_rcnn_label in enumerate(labels):
                tmp = masks[idx]
                mask[np.where(tmp)] = mask_rcnn_label
            mask = mask.astype(np.float32)
            mask /= len(self.objectTypes)
            return np.expand_dims(mask, axis=2)


class LocalKeyPoints3DSensorThor(Sensor):
    """Sensor for Key Points of objects in iTHOR.

    Returns from a running IThorEnvironment instance, the current
    key points of objects corresponding to the agent's egocentric view.
    """
    def __init__(self,
                 objectTypes,
                 uuid="class_segmentation"):
        self.objectTypes = objectTypes
        self.sorted_objectTypes = sorted(list(objectTypes))
        observation_space = gym.spaces.Box(
            low=-10,
            high=10,
            shape=(len(objectTypes), 8, 3),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        key_points = []
        current_depth = env.current_depth
        current_depth = np.expand_dims(current_depth, axis=2)
        if not env.using_mask_rcnn:
            for objType in self.objectTypes:
                mask = env.get_mask_by_object_type(objType)
                points, depths = get_corners(mask, current_depth)
                points = torch.Tensor(np.array([points]))
                depths = torch.Tensor(np.array([depths]))
                points_3d = local_project_2d_points_to_3d([env.last_event.metadata], points, depths)
                key_points.append(points_3d.numpy()[0])
        else:
            output = env.get_mask_rcnn_result()
            labels = list(output["labels"].detach().cpu().numpy())
            masks = output["masks"].squeeze(1).detach().cpu().numpy()
            for objType in self.objectTypes:
                mask_rcnn_label = self.sorted_objectTypes.index(objType)
                if mask_rcnn_label in labels:
                    idx = labels.index(mask_rcnn_label)
                    mask = masks[idx]
                else:
                    mask = np.zeros((env.current_frame.shape[0], env.current_frame.shape[1]))
                points, depths = get_corners(mask, current_depth)
                points = torch.Tensor([points])
                depths = torch.Tensor([depths])
                points_3d = local_project_2d_points_to_3d([env.last_event.metadata], points, depths)
                key_points.append(points_3d.numpy()[0])

        key_points = np.array(key_points, dtype=np.float32)
        return key_points


class GlobalKeyPoints3DSensorThor(Sensor):
    """Sensor for Key Points of objects in iTHOR.

    Returns from a running IThorEnvironment instance, the current
    key points of objects corresponding to the agent's egocentric view.
    """
    def __init__(self,
                 objectTypes,
                 uuid="class_segmentation"):
        self.objectTypes = objectTypes
        self.sorted_objectTypes = sorted(list(objectTypes))
        observation_space = gym.spaces.Box(
            low=-10,
            high=10,
            shape=(len(objectTypes), 8, 3),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        key_points = []
        current_depth = env.current_depth
        current_depth = np.expand_dims(current_depth, axis=2)
        if not env.using_mask_rcnn:
            for objType in self.objectTypes:
                mask = env.get_mask_by_object_type(objType)
                points, depths = get_corners(mask, current_depth)
                points = torch.Tensor(np.array([points]))
                depths = torch.Tensor(np.array([depths]))
                points_3d = project_2d_points_to_3d([env.last_event.metadata], points, depths)
                key_points.append(points_3d.numpy()[0])
        else:
            output = env.get_mask_rcnn_result()
            labels = list(output["labels"].detach().cpu().numpy())
            masks = output["masks"].squeeze(1).detach().cpu().numpy()
            for objType in self.objectTypes:
                mask_rcnn_label = self.sorted_objectTypes.index(objType)
                if mask_rcnn_label in labels:
                    idx = labels.index(mask_rcnn_label)
                    mask = masks[idx]
                else:
                    mask = np.zeros((env.current_frame.shape[0], env.current_frame.shape[1]))
                points, depths = get_corners(mask, current_depth)
                points = torch.Tensor([points])
                depths = torch.Tensor([depths])
                points_3d = project_2d_points_to_3d([env.last_event.metadata], points, depths)
                key_points.append(points_3d.numpy()[0])
        key_points = np.array(key_points, dtype=np.float32)
        return key_points


class GlobalObjPoseSensorThor(Sensor):
    def __init__(self,
                 objectTypes,
                 uuid="object_pose"):
        self.objectTypes = objectTypes
        observation_space = gym.spaces.Box(
            low=-360,
            high=360,
            shape=(len(objectTypes), 6),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        vis_objects = env.visible_objects()
        vis_objects_type = [ele["objectType"] for ele in vis_objects]
        obj_pose = []
        for objType in self.objectTypes:
            if objType in vis_objects_type:
                idx = vis_objects_type.index(objType)
                pose = [vis_objects[idx]["position"]["x"],
                        vis_objects[idx]["position"]["y"],
                        vis_objects[idx]["position"]["z"],
                        vis_objects[idx]["rotation"]["x"],
                        vis_objects[idx]["rotation"]["y"],
                        vis_objects[idx]["rotation"]["z"]]
                obj_pose.append(pose)
            else:
                obj_pose.append([0, 0, 0, 0, 0, 0])
        return np.array(obj_pose, dtype=np.float32)


class GlobalObjUpdateMaskSensorThor(Sensor):
    def __init__(self,
                 objectTypes,
                 uuid="object_update_mask"):
        self.objectTypes = objectTypes
        observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(objectTypes), 1),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        vis_objects = env.visible_objects()
        vis_objects_type = [ele["objectType"] for ele in vis_objects]
        update_mask = []
        for objType in self.objectTypes:
            if objType in vis_objects_type:
                update_mask.append(1)
            else:
                update_mask.append(0)
        return np.array(update_mask, dtype=np.float32)


class GlobalObjActionMaskSensorThor(Sensor):
    def __init__(self,
                 objectTypes,
                 uuid="object_action_mask"):
        self.objectTypes = objectTypes
        observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(objectTypes), 1),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        obj = env.moveable_closest_obj_by_types(self.objectTypes)
        update_mask = [0] * len(self.objectTypes)
        if not isinstance(obj, type(None)):
            idx = self.objectTypes.index(obj["objectType"])
            update_mask[idx] = 1
        return np.array(update_mask, dtype=np.float32)


class GlobalAgentPoseSensorThor(Sensor):
    def __init__(self,
                 uuid="agent_pose"):
        observation_space = gym.spaces.Box(
            low=-360,
            high=360,
            shape=(6,),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        agent_pose = [env.last_event.metadata["cameraPosition"]["x"],
                      env.last_event.metadata["cameraPosition"]["y"],
                      env.last_event.metadata["cameraPosition"]["z"],
                      env.last_event.metadata["agent"]["cameraHorizon"],
                      env.last_event.metadata["agent"]["rotation"]["y"],
                      0]
        return np.array(agent_pose, dtype=np.float32)

class MissingActionSensor(Sensor):
    def __init__(
            self,
            nactions: int,
            uuid: str = "missing_action",
            **kwargs: Any
    ) -> None:
        self.nactions = nactions
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Tuple:
        return gym.spaces.Discrete(self.nactions + 1)


    def get_observation(
            self, env: IThorEnvironment, task: Optional[ObjectPlacementTask], *args: Any, **kwargs: Any
    ) -> Any:
        missing_action = task.task_info["missing_action"]
        return missing_action

class MissingActionVectorSensor(Sensor):
    def __init__(
            self,
            nactions: int,
            uuid: str = "missing_action",
            **kwargs: Any
    ) -> None:
        self.nactions = nactions
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Tuple:
        return gym.spaces.Discrete(self.nactions + 1)

    def get_observation(
            self, env: IThorEnvironment, task: Optional[ObjectPlacementTask], *args: Any, **kwargs: Any
    ) -> Any:
        missing_action = task.task_info["missing_action"]
        out = np.zeros(self.nactions + 1)
        for ma in missing_action:
            out[ma] = 1
        return out


class MissingActionVectorMaskSensor(Sensor):
    def __init__(
            self,
            uuid: str = "missing_action_mask",
            **kwargs: Any
    ) -> None:
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Tuple:
        return gym.spaces.Discrete(1)

    def get_observation(
            self, env: IThorEnvironment, task: Optional[ObjectPlacementTask], *args: Any, **kwargs: Any
    ) -> Any:
        out = np.zeros(1)
        out[0] = task.missing_action_mask
        return out

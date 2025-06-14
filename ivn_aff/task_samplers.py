import random, gzip, json, gym, torch
from typing import List, Dict, Optional, Any, Union, Tuple

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.cache_utils import str_to_pos_for_cache
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed

from ivn_aff.environment import IThorEnvironment
from ivn_aff.tasks_dataset import ObstaclesNavTask, ObjectPlacementTask
from allenact_plugins.ithor_plugin.ithor_util import (
    round_to_factor,
    include_object_data,
)


class ObstaclesNavDatasetTaskSampler(TaskSampler):
    def __init__(
            self,
            scenes: List[str],
            scene_directory: str,
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            rewards_config: Dict,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            loop_dataset: bool = True,
            shuffle_dataset: bool = True,
            allow_flipping=False,
            env_class=IThorEnvironment,
            **kwargs,
    ) -> None:
        self.seed: Optional[int] = None
        self.set_seed(seed)
        self.rewards_config = rewards_config
        self.env_args = env_args
        random.shuffle(scenes)
        self.scenes = scenes
        # self.scenes = random.sample(scenes,30)           #随机选择100个房间测试
        self.scene_directory = scene_directory
        self.shuffle_dataset: bool = shuffle_dataset
        self.episodes = {}
        self.wbb = 0  # 计数方式，当这一类房子中连续多个episode不能找到障碍物对侧点，则跳过该房子
        #测试时候加载
        if not loop_dataset:
            self.episodes = {
                scene: self.load_dataset(
                    self.env_args['prior_dataset'],scene, scene_directory + "/episodes"
                )
                for scene in scenes
            }
        self.env_class = env_class
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0
        self.num=0

        self._last_sampled_task: Optional[ObstaclesNavTask] = None


        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @staticmethod # test
    def load_dataset(dataset, scene: str, base_directory: str) -> List[Dict]:
        filename = (
            "/".join([base_directory, scene])
            if base_directory[-1] != "/"
            else "".join([base_directory, scene])
        )
        filename += ".json.gz"
        fin = gzip.GzipFile(filename, "r")
        json_bytes = fin.read()
        fin.close()
        json_str = json_bytes.decode("utf-8")
        data = json.loads(json_str)
        random.shuffle(data)

        #测试时候使用，每个episode挑选三个
        for num,episode in enumerate(data):
            name = episode['scene']
            if len(dataset[int(name)]['rooms'])<3 and episode['shortest_path_length']<2:
                data.pop(num)
            if len(dataset[int(name)]['rooms']) > 5 and len(episode['all_paths'][0])<3:
                data.pop(num)
        data=random.sample(data,10)
        return data
    
    # @staticmethod # train
    # def load_dataset(scene: str, base_directory: str) -> List[Dict]:
    #     filename = (
    #         "/".join([base_directory, scene])
    #         if base_directory[-1] != "/"
    #         else "".join([base_directory, scene])
    #     )
    #     filename += ".json.gz"
    #     fin = gzip.GzipFile(filename, "r")
    #     json_bytes = fin.read()
    #     fin.close()
    #     json_str = json_bytes.decode("utf-8")
    #     data = json.loads(json_str)
    #     random.shuffle(data)

    #     #测试时候使用，每个episode挑选三个
    #     return data

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[ObstaclesNavTask]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObstaclesNavTask]:
        
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None
        if self.scene_index == len(self.scenes):
            print('遍历了一趟数据集')
            # self.max_tasks = None

        if self.scenes[self.scene_index] not in self.episodes:
            # print('加载', self.scenes[self.scene_index])
            self.episodes = {
                self.scenes[self.scene_index]: self.load_dataset(
                    self.scenes[self.scene_index], self.scene_directory + "/episodes"
                )
            }
        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # 向里面新添加数据集
            if self.scenes[self.scene_index] not in self.episodes:
                # print('加载',self.scenes[self.scene_index])
                self.episodes = {
                    self.scenes[self.scene_index]: self.load_dataset(
                        self.scenes[self.scene_index], self.scene_directory + "/episodes"
                    )
                }
                self.wbb = 0  # 换了新房间，重新开始计数，直到连续7个episode不能能放置成功
            # shuffle the new list of episodes to train on
            if self.shuffle_dataset:
                random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0

        # print(self.scene_index, self.scenes, self.episode_index)
        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is not None:
            # if scene.replace("_physics", "") != self.env.scene_name.replace(
            #        "_physics", ""
            # ):
            self.env.reset(scene_index=scene, filtered_objects=[])
        else:
            self.env = self._create_environment()
            self.env.reset(scene_index=scene, filtered_objects=[])

        def to_pos(s):
            if isinstance(s, (Dict, Tuple)):
                return s
            if isinstance(s, float):
                return {"x": 0, "y": s, "z": 0}
            return str_to_pos_for_cache(s)

        for k in ["initial_position", "initial_orientation", "target_position"]:
            episode[k] = to_pos(episode[k])
        # print('episode',episode.keys())
        episode['all_node_paths'][0].pop(0)
        task_info = {
            "scene": scene,
            "initial_position": episode["initial_position"],
            "initial_orientation": episode["initial_orientation"],
            "target": episode["target_position"],
            "distance_to_target": episode["shortest_path_length"],
            "id": episode["id"],
            "spawn_objects": episode["spawn_objects"],
            "obstacles_types": episode["obstacle_types"],
            "all_paths": episode['all_paths'],
            "all_node_paths": episode['all_node_paths'],
        }



        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1

        if not self.env.teleport(
                pose=episode["initial_position"],
                rotation=round_to_factor(episode["initial_orientation"]['y'], 90) % 360, horizon=30
        ):
            return self.next_task()

        for obj in episode["spawn_objects"]:
            # if not self.env.spawn_obj(obj):
            # if random.choice([0,1]) == 0:
            if not self.env.spawn_proc_obj(obj):
                return self.next_task()
            # dist = self.env.distance_to_target(episode['target_position'])
            # if dist == -1:
            #     # print('blocked')
            #     self.env.controller.step(
            #         action='DisableObject', objectId=obj['objectId'])

        # print(episode['spawn_objects'])
        # print('任务所在场景++++++++++障碍物数量',task_info['id'],len(task_info["spawn_objects"]))
        # 训练交互器，将agent放置在物体旁的目标对侧
        # spawn_objects = task_info["spawn_objects"]
        # random.shuffle(spawn_objects)
        # spawned = False
        # print('障碍物放置完成')
        # spawn_object = spawn_objects[0]
        # for spawn_object in spawn_objects:
        #     if spawn_object['pickupable']:
        #         # 放置物体
        #         if not self.env.spawn_proc_obj(spawn_object):
        #             continue
        #         # print('有物体')
        #         task_info['spawn_object'] = spawn_object
        #         # 判断当前节点目标
        #         obj_pos = spawn_object['position']
        #         dist_to_nodes = [pow((obj_pos['x'] - node['x']), 2) + pow((obj_pos['x'] - node['x']), 2) for node in
        #                          episode['all_node_paths'][0]]
        #         node_index = dist_to_nodes.index(min(dist_to_nodes))
        #         target = episode['all_node_paths'][0][node_index]
        #         task_info['target'] = target  # 交互时当前目标点
        #         # print(task_info['scene'])
        #         interactable_positions = self.env._interactable_positions_cache.get(
        #             scene_name=task_info['scene'], obj=spawn_object, controller=self.env.controller,
        #         )
        #         # 保留xyz坐标
        #         interactable_positions = [{'x': pos['x'], 'y': pos['y'], 'z': pos['z']} for pos in
        #                                   interactable_positions if
        #                                   pos['standing'] is True and pos['horizon'] == 30]
        #         # if len(interactable_positions)==0:
        #         #     print('没有可交互点')
        #         # 去除目标侧边的点
        #         interactable_positions = [pos for pos in interactable_positions if
        #                                   (pos['x'] - obj_pos['x']) * (target['x'] - obj_pos['x']) < 0 and (
        #                                       pos['z'] - obj_pos['z']) * (target['z'] - obj_pos['z']) < 0]
        #         dis_to_target = [self.env.position_dist(dict(x=pos['x'], z=pos['z'], y=0), target, ignore_y=True) for
        #                          pos in
        #                          interactable_positions]
        #
        #         if len(dis_to_target) > 0:
        #             position = interactable_positions[dis_to_target.index(min(dis_to_target))]
        #             rot = (self.env.azimuthAngle(obj_pos['x'], obj_pos['z'], position['x'], position['z']) + 180) % 360
        #             # print(rot)
        #             self.env.controller.step({'action': "Teleport",
        #                                       'position': position,
        #                                       'rotation': dict(x=0, y=rot, z=0),
        #                                       'horizon': 30})
        #             # print(position, obj_pos, target)
        #             if self.env.last_event.metadata['lastActionSuccess']:
        #                 spawned = True
        #                 break
        #             else:
        #                 # 移除放置不成功的物体
        #                 self.env.controller.step(
        #                     action='DisableObject', objectId=spawn_object['objectId'],
        #                 )
        #                 # else:
        #                 #     print('没有点')
        # if not spawned:
        #     # print('无法成功布置环境')
        #     self.wbb += 1  # 由于不能spawn成功导致的失败,这一类房间连续10个episode不能spawn，跳过该房间
        #     return self.next_task()
        # print('采样成功')

        self.env.initialize(0.25, renderObjectImage=True, renderDepthImage=True)

        self._last_sampled_task = ObstaclesNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )
        if torch.isinf(torch.from_numpy(self._last_sampled_task.get_observations()['rgb'])).any():
            print('房间编号',episode["id"])
            return self.next_task()
        if torch.isinf(torch.from_numpy(self._last_sampled_task.get_observations()['depth'])).any():
            print('房间编号',episode["id"])
            return self.next_task()


        # if torch.isinf(torch.from_numpy(task.get_observations()['rgb'])).any():
        #     print('房间编号', task.task_info["id"])
        # if torch.isinf(torch.from_numpy(task.get_observations()['depth'])).any():
        #     print('房间编号', task.task_info["id"])


        # print('成功布置环境')
        # if self._last_sampled_task.begin_dist==-1:
        #     return self.next_task()
        self.wbb = 0
        self.num += 1

        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks


class ObjectPlacementDatasetTaskSampler(TaskSampler):
    def __init__(
            self,
            scenes: List[str],
            scene_directory: str,
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            rewards_config: Dict,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            loop_dataset: bool = True,
            shuffle_dataset: bool = True,
            allow_flipping=False,
            env_class=IThorEnvironment,
            **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.shuffle_dataset: bool = shuffle_dataset
        # self.episodes = {
        #     scene: self.load_dataset(
        #         scene, scene_directory + "/episodes"
        #     )
        #     for scene in scenes
        # }
        self.env_class = env_class
        self.scene_directory = scene_directory
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[ObjectPlacementTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @staticmethod
    def load_dataset(scene: str, base_directory: str) -> List[Dict]:
        filename = (
            "/".join([base_directory, scene])
            if base_directory[-1] != "/"
            else "".join([base_directory, scene])
        )
        filename += ".json.gz"
        fin = gzip.GzipFile(filename, "r")
        json_bytes = fin.read()
        fin.close()
        json_str = json_bytes.decode("utf-8")
        data = json.loads(json_str)
        random.shuffle(data)
        return data

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[ObjectPlacementTask]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectPlacementTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None
        self.episodes = {
            self.scenes[self.scene_index]: self.load_dataset(
                self.scenes[self.scene_index], self.scene_directory + "/episodes"
            )
        }

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            if self.shuffle_dataset:
                random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0

        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is not None:
            # if scene.replace("_physics", "") != self.env.scene_name.replace(
            #        "_physics", ""
            # ):
            self.env.reset(scene_name=scene, filtered_objects=[])
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene, filtered_objects=[])

        def to_pos(s):
            if isinstance(s, (Dict, Tuple)):
                return s
            if isinstance(s, float):
                return {"x": 0, "y": s, "z": 0}
            return str_to_pos_for_cache(s)

        for k in ["initial_position", "initial_orientation", "target_position"]:
            episode[k] = to_pos(episode[k])

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1

        if not self.env.teleport(
                pose=episode["initial_position"],
                rotation=episode["initial_orientation"],
                horizon=episode["initial_horizon"]
        ):
            return self.next_task()

        for obj in episode["spawn_objects"]:
            if not "TargetCircle" in obj["objectType"]:
                if not self.env.spawn_obj(obj):
                    return self.next_task()
            else:
                if not self.env.spawn_target_circle(obj["random_seed"]):
                    return self.next_task()

        self.env.initialize(0.25, renderObjectImage=True, renderDepthImage=True)

        task_info = {
            "scene": scene,
            "initial_position": episode["initial_position"],
            "initial_orientation": episode["initial_orientation"],
            "target": episode["target_position"],
            "shortest_path": episode["shortest_path"],
            "distance_to_target": episode["shortest_path_length"],
            "id": episode["id"],
            "spawn_objects": episode["spawn_objects"],
            "obstacles_types": episode["obstacles_types"],
            "target_type": "TargetCircle",
            "object_type": episode["spawn_objects"][1]["objectType"],
        }

        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self._last_sampled_task = ObjectPlacementTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

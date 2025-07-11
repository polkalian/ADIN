"""A wrapper for engaging with the THOR environment."""

import prior
from procthor.utils.types import  Vector3
from procthor.constants import FLOOR_Y, OPENNESS_RANDOMIZATIONS
import copy
import functools
import math
import random
import typing
import warnings
import os
import re
import lru
from typing import Tuple, Dict, List, Set, Union, Any, Optional, Mapping, Sequence

import ai2thor.server
import ai2thor.fifo_server
import networkx as nx
import numpy as np
from ai2thor.controller import Controller
from ai2thor.util import metrics

from allenact.utils.cache_utils import (
    DynamicDistanceCache,
    pos_to_str_for_cache,
    str_to_pos_for_cache,
)
from allenact.utils.system import get_logger

from ivn_aff.constants import VISIBILITY_DISTANCE, FOV
from allenact_plugins.ithor_plugin.ithor_util import round_to_factor, include_object_data

import torch
from procthor.utils.types import  Vector3
from procthor.constants import FLOOR_Y, OPENNESS_RANDOMIZATIONS
# from ivn_proc.utils.train_mask_rcnn import get_model_instance_segmentation


class ObjectInteractablePostionsCache:
    def __init__(self, max_size: int = 20000, ndigits=2):
        self._key_to_positions = lru.LRU(size=max_size)

        self.ndigits = ndigits
        self.max_size = max_size

    def _get_key(self, scene_name: str, obj: Dict[str, Any]):
        p = obj["position"]
        return (
            scene_name,
            obj["type"] if "type" in obj else obj["objectType"],
            round(p["x"], self.ndigits),
            round(p["y"], self.ndigits),
            round(p["z"], self.ndigits),
        )

    def get(
        self,
        scene_name: str,
        obj: Dict[str, Any],
        controller: ai2thor.controller.Controller,
        reachable_positions: Optional[Sequence[Dict[str, float]]] = None,
        force_cache_refresh: bool = False,
    ) -> List[Dict[str, Union[float, int, bool]]]:
        scene_name = scene_name.replace("_physics", "")
        obj_key = self._get_key(scene_name=scene_name, obj=obj)

        if force_cache_refresh or obj_key not in self._key_to_positions:
            with include_object_data(controller):
                metadata = controller.last_event.metadata

            # cur_scene_name = metadata["sceneName"].replace("_physics", "")
            # assert (
            #     scene_name == cur_scene_name
            # ), f"Scene names must match when filling a cache miss ({scene_name} != {cur_scene_name})."

            obj_in_scene = next(
                (o for o in metadata["objects"] if o["name"] == obj["name"]), None,
            )
            if obj_in_scene is None:
                raise RuntimeError(
                    f"Object with name {obj['name']} must be in the scene when filling a cache miss"
                )

            desired_pos = obj["position"]
            desired_rot = obj["rotation"]

            cur_pos = obj_in_scene["position"]
            cur_rot = obj_in_scene["rotation"]

            # should_teleport = (
            #     IThorEnvironment.position_dist(desired_pos, cur_pos) >= 1e-3
            #     or IThorEnvironment.rotation_dist(desired_rot, cur_rot) >= 1
            # )
            should_teleport=False
            object_held = obj_in_scene["isPickedUp"]
            physics_was_unpaused = controller.last_event.metadata.get(
                "physicsAutoSimulation", True
            )
            if should_teleport:
                if object_held:
                    if not hand_in_initial_position(controller=controller):
                        raise NotImplementedError

                    if physics_was_unpaused:
                        controller.step("PausePhysicsAutoSim")
                        assert controller.last_event.metadata["lastActionSuccess"]

                event = controller.step(
                    "TeleportObject",
                    objectId=obj_in_scene["objectId"],
                    rotation=desired_rot,
                    **desired_pos,
                    forceAction=True,
                    allowTeleportOutOfHand=True,
                    forceKinematic=True,
                )
                assert event.metadata["lastActionSuccess"]

            metadata = controller.step(
                action="GetInteractablePoses",
                objectId=obj["objectId"],
                positions=reachable_positions,
            ).metadata
            if not metadata["lastActionSuccess"]:
                print(metadata["errorMessage"])
            assert metadata["lastActionSuccess"]
            self._key_to_positions[obj_key] = metadata["actionReturn"]

            if should_teleport:
                if object_held:
                    if hand_in_initial_position(controller=controller):
                        controller.step(
                            "PickupObject",
                            objectId=obj_in_scene["objectId"],
                            forceAction=True,
                        )
                        assert controller.last_event.metadata["lastActionSuccess"]

                        if physics_was_unpaused:
                            controller.step("UnpausePhysicsAutoSim")
                            assert controller.last_event.metadata["lastActionSuccess"]
                    else:
                        raise NotImplementedError
                else:
                    event = controller.step(
                        "TeleportObject",
                        objectId=obj_in_scene["objectId"],
                        rotation=cur_rot,
                        **cur_pos,
                        forceAction=True,
                    )
                    # assert event.metadata["lastActionSuccess"]
                    if not event.metadata["lastActionSuccess"]:
                        print('还原错误')

        return self._key_to_positions[obj_key]


class IThorEnvironment(object):
    """Wrapper for the ai2thor controller providing additional functionality
    and bookkeeping.

    See [here](https://ai2thor.allenai.org/documentation/installation) for comprehensive
     documentation on AI2-THOR.

    # Attributes

    controller : The ai2thor controller.
    """

    def __init__(
        self,
        prior_dataset,
        x_display: Optional[str] = None,
        docker_enabled: bool = False,
        local_thor_build: Optional[str] = None,
        visibility_distance: float = VISIBILITY_DISTANCE,
        fov: float = FOV,
        player_screen_width: int = 300,
        player_screen_height: int = 300,
        quality: str = "Very Low",
        restrict_to_initially_reachable_points: bool = False,
        make_agents_visible: bool = True,
        object_open_speed: float = 1.0,
        simplify_physics: bool = False,
        using_mask_rcnn: bool = False,
        mask_rcnn_dir: str = "",
        grid_size: float = 0.25,
        thor_commit_id: str = None,
    ) -> None:
        """Initializer.

        # Parameters

        x_display : The x display into which to launch ai2thor (possibly necessarily if you are running on a server
            without an attached display).
        docker_enabled : Whether or not to run thor in a docker container (useful on a server without an attached
            display so that you don't have to start an x display).
        local_thor_build : The path to a local build of ai2thor. This is probably not necessary for your use case
            and can be safely ignored.
        visibility_distance : The distance (in meters) at which objects, in the viewport of the agent,
            are considered visible by ai2thor and will have their "visible" flag be set to `True` in the metadata.
        fov : The agent's camera's field of view.
        player_screen_width : The width resolution (in pixels) of the images returned by ai2thor.
        player_screen_height : The height resolution (in pixels) of the images returned by ai2thor.
        quality : The quality at which to render. Possible quality settings can be found in
            `ai2thor._quality_settings.QUALITY_SETTINGS`.
        restrict_to_initially_reachable_points : Whether or not to restrict the agent to locations in ai2thor
            that were found to be (initially) reachable by the agent (i.e. reachable by the agent after resetting
            the scene). This can be useful if you want to ensure there are only a fixed set of locations where the
            agent can go.
        make_agents_visible : Whether or not the agent should be visible. Most noticable when there are multiple agents
            or when quality settings are high so that the agent casts a shadow.
        object_open_speed : How quickly objects should be opened. High speeds mean faster simulation but also mean
            that opening objects have a lot of kinetic energy and can, possibly, knock other objects away.
        simplify_physics : Whether or not to simplify physics when applicable. Currently this only simplies object
            interactions when opening drawers (when simplified, objects within a drawer do not slide around on
            their own when the drawer is opened or closed, instead they are effectively glued down).
        """

        self._start_player_screen_width = player_screen_width
        self._start_player_screen_height = player_screen_height
        self._local_thor_build = local_thor_build
        self.x_display = x_display
        self.controller: Optional[Controller] = None
        self._started = False
        self._quality = quality

        self.thor_commit_id = thor_commit_id

        self._initially_reachable_points: Optional[List[Dict]] = None
        self._initially_reachable_points_set: Optional[Set[Tuple[float, float]]] = None
        self._move_mag: Optional[float] = None
        self._grid_size: Optional[float] = None
        self._visibility_distance = visibility_distance
        self._fov = fov
        self.restrict_to_initially_reachable_points = (
            restrict_to_initially_reachable_points
        )
        self.make_agents_visible = make_agents_visible
        self.object_open_speed = object_open_speed
        self._always_return_visible_range = False
        self.simplify_physics = simplify_physics
        self.dataset = prior_dataset

        self.start(None, move_mag=grid_size)
        
        # noinspection PyTypeHints
        self.controller.docker_enabled = docker_enabled  # type: ignore

        #设置缓存，加快获取速度
        self.distance_cache = DynamicDistanceCache(rounding=1)

        self.last_frame_cache = None
        self.last_depth_cache = None

        self.counter = 0
        self.obstacle_ids = []
        self.seg_ids = []
        self.task_info = {}

        self.using_mask_rcnn = using_mask_rcnn
        if using_mask_rcnn and os.path.isfile(mask_rcnn_dir):
            gpu_id = int(x_display.split(".")[-1])
            loc = 'cuda:{}'.format(gpu_id)
            checkpoint = torch.load(mask_rcnn_dir, map_location=loc)
            self.mask_rcnn_model = get_model_instance_segmentation(21).cuda(gpu_id)
            self.mask_rcnn_model.load_state_dict(checkpoint["model"])
            self.mask_rcnn_model.eval()
            self.mask_rcnn_gpu_id = gpu_id
        else:
            self.mask_rcnn_model = None

        self._interactable_positions_cache = ObjectInteractablePostionsCache()

    @property
    def held_object(self) -> Optional[Dict[str, Any]]:
        """Return the data corresponding to the object held by the agent (if
        any)."""
        with include_object_data(self.controller):
            metadata = self.controller.last_event.metadata

            if len(metadata["inventoryObjects"]) == 0:
                return None

            assert len(metadata["inventoryObjects"]) <= 1

            held_obj_id = metadata["inventoryObjects"][0]["objectId"]
            return next(o for o in metadata["objects"] if o["objectId"] == held_obj_id)

    @property
    def scene_name(self) -> str:
        """Current ai2thor scene."""
        return self.controller.last_event.metadata["sceneName"]

    @property
    def current_frame(self) -> np.ndarray:
        """Returns rgb image corresponding to the agent's egocentric view."""
        return self.controller.last_event.frame

    @property
    def current_depth(self) -> np.ndarray:
        """Returns depth image corresponding to the agent's egocentric view."""
        return self.controller.last_event.depth_frame

    @property
    def last_frame(self) -> np.ndarray:
        if isinstance(self.last_frame_cache, type(None)):
            self.last_frame_cache = self.current_frame
        return self.last_frame_cache

    @property
    def last_depth(self) -> np.ndarray:
        if isinstance(self.last_depth_cache, type(None)):
            self.last_depth_cache = self.current_depth
        return self.last_depth_cache

    @property
    def current_instance_segmentation_frame(self) -> np.ndarray:
        """Returns instance segmentation frame corresponding to the agent's egocentric view."""
        return self.controller.last_event.instance_segmentation_frame

    @property
    def current_instance_masks(self):
        return self.controller.last_event.instance_masks

    @property
    def last_event(self) -> ai2thor.server.Event:
        """Last event returned by the controller."""
        return self.controller.last_event

    @property
    def started(self) -> bool:
        """Has the ai2thor controller been started."""
        return self._started

    @property
    def last_action(self) -> str:
        """Last action, as a string, taken by the agent."""
        return self.controller.last_event.metadata["lastAction"]

    @last_action.setter
    def last_action(self, value: str) -> None:
        """Set the last action taken by the agent.

        Doing this is rewriting history, be careful.
        """
        self.controller.last_event.metadata["lastAction"] = value

    @property
    def last_action_success(self) -> bool:
        """Was the last action taken by the agent a success?"""
        return self.controller.last_event.metadata["lastActionSuccess"]

    @last_action_success.setter
    def last_action_success(self, value: bool) -> None:
        """Set whether or not the last action taken by the agent was a success.

        Doing this is rewriting history, be careful.
        """
        self.controller.last_event.metadata["lastActionSuccess"] = value

    @property
    def last_action_return(self) -> Any:
        """Get the value returned by the last action (if applicable).

        For an example of an action that returns a value, see
        `"GetReachablePositions"`.
        """
        return self.controller.last_event.metadata["actionReturn"]

    @last_action_return.setter
    def last_action_return(self, value: Any) -> None:
        """Set the value returned by the last action.

        Doing this is rewriting history, be careful.
        """
        self.controller.last_event.metadata["actionReturn"] = value

    def start(
        self, scene_name: Optional[str], move_mag: float = 0.25, **kwargs,
    ) -> None:
        """Starts the ai2thor controller if it was previously stopped.

        After starting, `reset` will be called with the scene name and move magnitude.

        # Parameters

        scene_name : The scene to load.
        move_mag : The amount of distance the agent moves in a single `MoveAhead` step.
        kwargs : additional kwargs, passed to reset.
        """
        if self._started:
            raise RuntimeError(
                "Trying to start the environment but it is already started."
            )

        def create_controller():
            return Controller(
                x_display=self.x_display,
                player_screen_width=self._start_player_screen_width,
                player_screen_height=self._start_player_screen_height,
                local_executable_path=self._local_thor_build,
                quality=self._quality,
                fastActionEmit=True,
                server_class=ai2thor.fifo_server.FifoServer,
                commit_id=self.thor_commit_id,
            )

        self.controller = create_controller()

        if (
            self._start_player_screen_height,
            self._start_player_screen_width,
        ) != self.current_frame.shape[:2]:
            self.controller.step(
                {
                    "action": "ChangeResolution",
                    "x": self._start_player_screen_width,
                    "y": self._start_player_screen_height,
                }
            )

        self._started = True
        self.reset(scene_index='0', move_mag=move_mag, **kwargs)

    def stop(self) -> None:
        """Stops the ai2thor controller."""
        try:
            self.controller.stop()
        except Exception as e:
            warnings.warn(str(e))
        finally:
            self._started = False

    def initialize(self, move_mag: float = 0.25, **kwargs) -> None:
        self._move_mag = move_mag
        self._grid_size = self._move_mag

        self.controller.step(
            {
                "action": "Initialize",
                "gridSize": self._grid_size,
                "visibilityDistance": self._visibility_distance,
                "fov": self._fov,
                "makeAgentsVisible": self.make_agents_visible,
                "alwaysReturnVisibleRange": self._always_return_visible_range,
                "fastActionEmit": True,
                **kwargs,
            }
        )

    def reset(
        self, scene_index: Optional[int], move_mag: float = 0.25, **kwargs,
    ):
        """Resets the ai2thor in a new scene.

        Resets ai2thor into a new scene and initializes the scene/agents with
        prespecified settings (e.g. move magnitude).

        # Parameters

        scene_name : The scene to load.
        move_mag : The amount of distance the agent moves in a single `MoveAhead` step.
        kwargs : additional kwargs, passed to the controller "Initialize" action.
        """
        self._move_mag = move_mag
        self._grid_size = self._move_mag

        # if scene_name is None:
        #     scene_name = self.controller.last_event.metadata["sceneName"]
        # scene_index=re.findall("\d+",scene_index)[0]
        scene_index="".join(list(filter(str.isdigit, scene_index)))
        house = self.dataset[int(scene_index)]
        self.controller.reset(house)

        self.initialize(move_mag, **kwargs)

        if self.object_open_speed != 1.0:
            self.controller.step(
                {"action": "ChangeOpenSpeed", "x": self.object_open_speed}
            )

        self._initially_reachable_points = None
        self._initially_reachable_points_set = None
        self.controller.step({"action": "GetReachablePositions"})
        if not self.controller.last_event.metadata["lastActionSuccess"]:
            warnings.warn(
                "Error when getting reachable points: {}".format(
                    self.controller.last_event.metadata["errorMessage"]
                )
            )
        self._initially_reachable_points = self.last_action_return

        self.counter = 0

    def path_from_point_to_point(
        self, position: Dict[str, float], target: Dict[str, float]
    ) -> Optional[List[Dict[str, float]]]:
        try:
            return self.controller.step(
                action="GetShortestPathToPoint",
                position=Vector3(x=position['x'], y=position['y'], z=position['z']),
                target=Vector3(x=target['x'], y=target['y'], z=target['z']),
                # renderImage=False
            ).metadata["actionReturn"]["corners"]
        except:
            get_logger().debug(
                "Failed to find path for {} in {}. Start point {}, agent state {}.".format(
                    target,
                    self.controller.last_event.metadata["sceneName"],
                    position,
                    self.agent_state(),
                )
            )
            print("no path")
            return None

    def distance_from_point_to_point(
        self, position: Dict[str, float], target: Dict[str, float]
    ) -> float:
        path = self.path_from_point_to_point(position, target)
        is_arrived = position['x'] == target['x'] and position['z'] == target['z']
        if is_arrived:
            return 0.0
        if path:
            return metrics.path_distance(path)
        return -1.0

    def object_distance_to_point(self, scene_name, obj_idx, target):
        return self.distance_cache.find_distance(
            scene_name,
            self.controller.last_event.metadata["objects"][obj_idx]["position"],
            target,
            self.distance_from_point_to_point,
        )

    def distance_to_point(self, scene_name, target: Dict[str, float]) -> float:
        """Minimal geodesic distance to end point from agent's current
        location.

        It might return -1.0 for unreachable targets.
        """
        return self.distance_cache.find_distance(
            scene_name,
            self.controller.last_event.metadata["agent"]["position"],
            target,
            self.distance_from_point_to_point,
        )

    def random_distance_to_point(self, scene_name, position,target: Dict[str, float]) -> float:
        """Minimal geodesic distance to end point from agent's current
        location.

        It might return -1.0 for unreachable targets.
        """
        return self.distance_cache.find_distance(
            scene_name,
            position,
            target,
            self.distance_from_point_to_point,
        )

    def stand_pos_to_graph(self, stand_pos, grid):
        G = nx.Graph()
        for stand in stand_pos:
            stand_l = (stand[0] - grid, stand[1])
            stand_r = (stand[0] + grid, stand[1])
            stand_f = (stand[0], stand[1] - grid)
            stand_b = (stand[0], stand[1] + grid)
            if stand_l in stand_pos:
                G.add_edge(stand, stand_l)
            if stand_r in stand_pos:
                G.add_edge(stand, stand_r)
            if stand_f in stand_pos:
                G.add_edge(stand, stand_f)
            if stand_b in stand_pos:
                G.add_edge(stand, stand_b)

        return G

    def distance_to_target(self, target: Dict[str, float]) -> float:
        self.controller.step({"action": "GetReachablePositions"})
        stand_pos = self.controller.last_event.metadata['actionReturn']
        if not isinstance(stand_pos,list):
            return -1
        stand_pos_ = []
        for stand in stand_pos:
            stand_ = (stand['x'], stand['z'])
            stand_pos_.append(stand_)
        graph = self.stand_pos_to_graph(stand_pos_, 0.25)
        agent_pos = (self.controller.last_event.metadata["agent"]["position"]['x'], self.controller.last_event.metadata["agent"]["position"]['z'])
        target_pos = (target['x'], target['z'])
        try:
            return nx.shortest_path_length(graph, agent_pos, target_pos) * 0.25
        except:
            return -1

    def path_to_target(self, target: Dict[str, float]):
        self.controller.step({"action": "GetReachablePositions"})
        stand_pos = self.controller.last_event.metadata['actionReturn']
        if not isinstance(stand_pos,list):
            return -1
        stand_pos_ = []
        for stand in stand_pos:
            stand_ = (stand['x'], stand['z'])
            stand_pos_.append(stand_)
        graph = self.stand_pos_to_graph(stand_pos_, 0.25)
        agent_pos = (self.controller.last_event.metadata["agent"]["position"]['x'], self.controller.last_event.metadata["agent"]["position"]['z'])
        target_pos = (target['x'], target['z'])
        try:
            return nx.shortest_path(graph, agent_pos, target_pos)
        except:
            return -1

    def agent_state(self) -> Dict:
        """Return agent position, rotation and horizon."""
        agent_meta = self.last_event.metadata["agent"]
        return {
            **{k: float(v) for k, v in agent_meta["position"].items()},
            "rotation": {k: float(v) for k, v in agent_meta["rotation"].items()},
            "horizon": round(float(agent_meta["cameraHorizon"]), 1),
        }

    def teleport(
        self, pose: Dict[str, float], rotation: Dict[str, float], horizon: float = 0.0
    ):
        e = self.controller.step(
            action="Teleport",
            x=pose["x"],
            y=pose["y"],
            z=pose["z"],
            rotation=rotation,
            horizon=horizon,
        )
        return e.metadata["lastActionSuccess"]

    def spawn_obj(self, obj):
        # e = self.controller.step(
        #     action="CreateObjectOnFloor",
        #     objectType=obj["objectType"],
        #     objectVariation=obj["objectVariation"],
        #     x=obj["position"]["x"],
        #     z=obj["position"]["z"],
        #     rotation=obj["rotation"],
        #     randomizeObjectAppearance=False
        # )
        e = self.controller.step(action="PlaceObjectAtPoint", objectId=obj['objectId'], position=obj['position'])
        return e.metadata["lastActionSuccess"]
    def spawn_ourobj(self,obj):
        e = self.controller.step(action="PlaceObjectAtPoint", objectId=obj['objectId'], position=obj['position'])
        return e.metadata["lastActionSuccess"]

    def spawn_proc_obj(self, obj):
        e = self.controller.step(
            action="SpawnAsset",
            assetId=obj['assetId'],
            generatedId=obj['objectId'],
            position=Vector3(x=obj['position']['x'], y=FLOOR_Y, z=obj['position']['z']),
            renderImage=False,
        )
        self.obstacle_ids.append(obj["objectId"])
        return e.metadata["lastActionSuccess"]

    def spawn_target_circle(self, seed):
        obj = self.get_objects_by_name("Floor")
        e = self.controller.step(
            action="SpawnTargetCircle",
            objectId=obj["objectId"],
            anywhere=True,
            objectVariation=2,
            randomSeed=seed
        )
        return e.metadata["lastActionSuccess"]

    def get_objects_by_name(self, object_name):
        objs = self.all_objects()
        for obj in objs:
            if object_name in obj["name"]:
                return obj
        return None

    def target_in_reachable_points(self, tget) -> bool:
        reachable_points = self.currently_reachable_points
        reachable_points = np.array([[ele["x"], ele["z"]] for ele in reachable_points])
        tget = np.array([tget["x"], tget["z"]])
        if tget in reachable_points:
            return True
        else:
            return False

    @property
    def moveable_closest_obj(self):
        objs = self.visible_objects()
        objs = [ele for ele in objs if ele["moveable"] or ele["pickupable"]]
        obj = None
        if len(objs) > 0:
            agent_pos = [self.last_event.metadata["agent"]["position"]["x"],
                         self.last_event.metadata["agent"]["position"]["z"]]
            dis = [(obj["position"]["x"] - agent_pos[0]) ** 2 +
                   (obj["position"]["z"] - agent_pos[1]) ** 2 for obj in objs]
            idx = np.argmin(dis)
            obj = objs[idx]
        return obj

    def get_objects_by_type(self, objectTypes):
        objs = self.all_objects()
        objs = [ele for ele in objs if ele["objectType"] in objectTypes]
        return objs

    def get_objects_and_idx_by_type(self, objectTypes):
        objs = self.all_objects()
        obj, idx = [], []
        for idy, ele in enumerate(objs):
            if ele["objectType"] in objectTypes:
                obj.append(ele)
                idx.append(idy)
                break
        return obj, idx

    def moveable_closest_obj_by_types(self, objectTypes):
        objs = self.visible_objects()
        # objs = self.all_objects()
        objs = [ele for ele in objs if ele["moveable"]]
        objs = [ele for ele in objs if ele["objectType"] in objectTypes]
        # print(len(objs))
        # objs = [ele for ele in objs if ele["receptacleObjectIds"] == [] or ele["receptacleObjectIds"] is None]
        obj = None
        if len(objs) > 0:
            agent_pos = [self.last_event.metadata["agent"]["position"]["x"],
                         self.last_event.metadata["agent"]["position"]["z"]]
            dis = [(obj["position"]["x"] - agent_pos[0]) ** 2 +
                   (obj["position"]["z"] - agent_pos[1]) ** 2 for obj in objs]
            idx = np.argmin(dis)
            # if dis[idx] <= 1.5:
            #     obj = objs[idx]
            # else:
            #     obj = None
            obj = objs[idx]
        return obj

    def pickupable_closest_obj_by_types(self, objectTypes):
        objs = self.visible_objects()
        # objs = self.all_objects()
        objs = [ele for ele in objs if ele["pickupable"]]
        objs = [ele for ele in objs if ele["objectType"] in objectTypes]
        # objs = [ele for ele in objs if ele["receptacleObjectIds"] == [] or ele["receptacleObjectIds"] is None]
        obj = None
        if len(objs) > 0:
            agent_pos = [self.last_event.metadata["agent"]["position"]["x"],
                         self.last_event.metadata["agent"]["position"]["z"]]
            dis = [(obj["position"]["x"] - agent_pos[0]) ** 2 +
                   (obj["position"]["z"] - agent_pos[1]) ** 2 for obj in objs]
            idx = np.argmin(dis)
            # if dis[idx] <= 1.5:
            #     obj = objs[idx]
            # else:
            #     obj = None
            obj = objs[idx]
        return obj

    def reachable_farthest_point(self):
        event = self.controller.step(action="GetReachablePositions")
        pos = event.metadata['actionReturn']
        # agent_pos = [self.last_event.metadata["agent"]["position"]["x"],
        #              self.last_event.metadata["agent"]["position"]["z"]]
        # point = None
        # if len(pos) > 0:
        #     dis = [(pos_["x"] - agent_pos[0]) ** 2 +
        #            (pos_["z"] - agent_pos[1]) ** 2 for pos_ in pos]
        #     idx = np.argmax(dis)
        #     point = pos[idx]
        try:
            random.shuffle(pos)
            point = pos[0]
        except:
            point = None
        return point

    def azimuthAngle(self, x1, y1, x2, y2):
        angle = 0.0;
        dx = x2 - x1
        dy = y2 - y1
        if x2 == x1:
            angle = math.pi / 2.0
            if y2 == y1:
                angle = 0.0
            elif y2 < y1:
                angle = 3.0 * math.pi / 2.0
        elif x2 > x1 and y2 > y1:
            angle = math.atan(dx / dy)
        elif x2 > x1 and y2 < y1:
            angle = math.pi / 2 + math.atan(-dy / dx)
        elif x2 < x1 and y2 < y1:
            angle = math.pi + math.atan(dx / dy)
        elif x2 < x1 and y2 > y1:
            angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
        return (angle * 180 / math.pi)

    def get_mask_by_object_type(self, objectType):
        mask = np.zeros((self.current_frame.shape[0], self.current_frame.shape[1]))
        objs = self.visible_objects()
        objsIds = [ele["objectId"] for ele in objs if ele["objectType"] == objectType]
        if len(objsIds) > 0:
            for Id in objsIds:
                if Id in self.last_event.instance_masks.keys():
                    mask = np.maximum(mask, self.last_event.instance_masks[Id])
        return mask

    def get_mask_rcnn_result(self):
        img = np.array(self.current_frame).astype(np.float)
        img = np.transpose(img, (2, 0, 1)) / 255.
        img = torch.Tensor(img).to(self.mask_rcnn_gpu_id)
        return self.mask_rcnn_model([img])[0]

    def get_masks_by_object_types(self, objectTypes):
        mask = np.ones((self.current_frame.shape[0], self.current_frame.shape[0])) * len(objectTypes)
        for i, objType in enumerate(objectTypes):
            tmp = self.get_mask_by_object_type(objType)
            mask[np.where(tmp)] = i
        mask = mask.astype(np.float32)
        mask /= len(objectTypes)
        return np.expand_dims(mask, axis=2)

    def teleport_agent_to(
        self,
        x: float,
        y: float,
        z: float,
        rotation: float,
        horizon: float,
        standing: Optional[bool] = None,
        force_action: bool = False,
        only_initially_reachable: Optional[bool] = None,
        verbose=True,
        ignore_y_diffs=False,
    ) -> None:
        """Helper function teleporting the agent to a given location."""
        if standing is None:
            standing = self.last_event.metadata.get(
                "isStanding", self.last_event.metadata["agent"].get("isStanding")
            )
        original_location = self.get_agent_location()
        target = {"x": x, "y": y, "z": z}
        if only_initially_reachable is None:
            only_initially_reachable = self.restrict_to_initially_reachable_points
        if only_initially_reachable:
            reachable_points = self.initially_reachable_points
            reachable = False
            for p in reachable_points:
                if self.position_dist(target, p, ignore_y=ignore_y_diffs) < 0.01:
                    reachable = True
                    break
            if not reachable:
                self.last_action = "Teleport"
                self.last_event.metadata[
                    "errorMessage"
                ] = "Target position was not initially reachable."
                self.last_action_success = False
                return
        self.controller.step(
            dict(
                action="TeleportFull",
                x=x,
                y=y,
                z=z,
                rotation={"x": 0.0, "y": rotation, "z": 0.0},
                horizon=horizon,
                standing=standing,
                forceAction=force_action,
            )
        )
        if not self.last_action_success:
            agent_location = self.get_agent_location()
            rot_diff = (
                agent_location["rotation"] - original_location["rotation"]
            ) % 360
            new_old_dist = self.position_dist(
                original_location, agent_location, ignore_y=ignore_y_diffs
            )
            if (
                self.position_dist(
                    original_location, agent_location, ignore_y=ignore_y_diffs
                )
                > 1e-2
                or min(rot_diff, 360 - rot_diff) > 1
            ):
                warnings.warn(
                    "Teleportation FAILED but agent still moved (position_dist {}, rot diff {})"
                    " (\nprevious location\n{}\ncurrent_location\n{}\n)".format(
                        new_old_dist, rot_diff, original_location, agent_location
                    )
                )
            return

        if force_action:
            assert self.last_action_success
            return

        agent_location = self.get_agent_location()
        rot_diff = (agent_location["rotation"] - rotation) % 360
        if (
            self.position_dist(agent_location, target, ignore_y=ignore_y_diffs) > 1e-2
            or min(rot_diff, 360 - rot_diff) > 1
        ):
            if only_initially_reachable:
                self._snap_agent_to_initially_reachable(verbose=False)
            if verbose:
                warnings.warn(
                    "Teleportation did not place agent"
                    " precisely where desired in scene {}"
                    " (\ndesired\n{}\nactual\n{}\n)"
                    " perhaps due to grid snapping."
                    " Action is considered failed but agent may have moved.".format(
                        self.scene_name,
                        {
                            "x": x,
                            "y": y,
                            "z": z,
                            "rotation": rotation,
                            "standing": standing,
                            "horizon": horizon,
                        },
                        agent_location,
                    )
                )
            self.last_action_success = False
        return

    def random_reachable_state(self, seed: int = None) -> Dict:
        """Returns a random reachable location in the scene."""
        if seed is not None:
            random.seed(seed)
        xyz = random.choice(self.currently_reachable_points)
        rotation = random.choice([0, 90, 180, 270])
        horizon = random.choice([0, 30, 60, 330])
        state = copy.copy(xyz)
        state["rotation"] = rotation
        state["horizon"] = horizon
        return state

    def randomize_agent_location(
        self, seed: int = None, partial_position: Optional[Dict[str, float]] = None
    ) -> Dict:
        """Teleports the agent to a random reachable location in the scene."""
        if partial_position is None:
            partial_position = {}
        k = 0
        state: Optional[Dict] = None

        while k == 0 or (not self.last_action_success and k < 10):
            state = self.random_reachable_state(seed=seed)
            self.teleport_agent_to(**{**state, **partial_position})
            k += 1

        if not self.last_action_success:
            warnings.warn(
                (
                    "Randomize agent location in scene {}"
                    " with seed {} and partial position {} failed in "
                    "10 attempts. Forcing the action."
                ).format(self.scene_name, seed, partial_position)
            )
            self.teleport_agent_to(**{**state, **partial_position}, force_action=True)  # type: ignore
            assert self.last_action_success

        assert state is not None
        return state

    def object_pixels_in_frame(
        self, object_id: str, hide_all: bool = True, hide_transparent: bool = False
    ) -> np.ndarray:
        """Return an mask for a given object in the agent's current view.

        # Parameters

        object_id : The id of the object.
        hide_all : Whether or not to hide all other objects in the scene before getting the mask.
        hide_transparent : Whether or not partially transparent objects are considered to occlude the object.

        # Returns

        A numpy array of the mask.
        """

        # Emphasizing an object turns it magenta and hides all other objects
        # from view, we can find where the hand object is on the screen by
        # emphasizing it and then scanning across the image for the magenta pixels.
        if hide_all:
            self.step({"action": "EmphasizeObject", "objectId": object_id})
        else:
            self.step({"action": "MaskObject", "objectId": object_id})
            if hide_transparent:
                self.step({"action": "HideTranslucentObjects"})
        # noinspection PyShadowingBuiltins
        filter = np.array([[[255, 0, 255]]])
        object_pixels = 1 * np.all(self.current_frame == filter, axis=2)
        if hide_all:
            self.step({"action": "UnemphasizeAll"})
        else:
            self.step({"action": "UnmaskObject", "objectId": object_id})
            if hide_transparent:
                self.step({"action": "UnhideAllObjects"})
        return object_pixels

    def object_pixels_on_grid(
        self,
        object_id: str,
        grid_shape: Tuple[int, int],
        hide_all: bool = True,
        hide_transparent: bool = False,
    ) -> np.ndarray:
        """Like `object_pixels_in_frame` but counts object pixels in a
        partitioning of the image."""

        def partition(n, num_parts):
            m = n // num_parts
            parts = [m] * num_parts
            num_extra = n % num_parts
            for k in range(num_extra):
                parts[k] += 1
            return parts

        object_pixels = self.object_pixels_in_frame(
            object_id=object_id, hide_all=hide_all, hide_transparent=hide_transparent
        )

        # Divide the current frame into a grid and count the number
        # of hand object pixels in each of the grid squares
        sums_in_blocks: List[List] = []
        frame_shape = self.current_frame.shape[:2]
        row_inds = np.cumsum([0] + partition(frame_shape[0], grid_shape[0]))
        col_inds = np.cumsum([0] + partition(frame_shape[1], grid_shape[1]))
        for i in range(len(row_inds) - 1):
            sums_in_blocks.append([])
            for j in range(len(col_inds) - 1):
                sums_in_blocks[i].append(
                    np.sum(
                        object_pixels[
                            row_inds[i] : row_inds[i + 1], col_inds[j] : col_inds[j + 1]
                        ]
                    )
                )
        return np.array(sums_in_blocks, dtype=np.float32)

    def object_in_hand(self):
        """Object metadata for the object in the agent's hand."""
        inv_objs = self.last_event.metadata["inventoryObjects"]
        if len(inv_objs) == 0:
            return None
        elif len(inv_objs) == 1:
            return self.get_object_by_id(
                self.last_event.metadata["inventoryObjects"][0]["objectId"]
            )
        else:
            raise AttributeError("Must be <= 1 inventory objects.")

    @property
    def initially_reachable_points(self) -> List[Dict[str, float]]:
        """List of {"x": x, "y": y, "z": z} locations in the scene that were
        reachable after initially resetting."""
        assert self._initially_reachable_points is not None
        return copy.deepcopy(self._initially_reachable_points)  # type:ignore

    @property
    def initially_reachable_points_set(self) -> Set[Tuple[float, float]]:
        """Set of (x,z) locations in the scene that were reachable after
        initially resetting."""
        if self._initially_reachable_points_set is None:
            self._initially_reachable_points_set = set()
            for p in self.initially_reachable_points:
                self._initially_reachable_points_set.add(
                    self._agent_location_to_tuple(p)
                )

        return self._initially_reachable_points_set

    @property
    def currently_reachable_points(self) -> List[Dict[str, float]]:
        """List of {"x": x, "y": y, "z": z} locations in the scene that are
        currently reachable."""
        self.step({"action": "GetReachablePositions"})
        return self.last_event.metadata["reachablePositions"]  # type:ignore

    def get_agent_location(self) -> Dict[str, Union[float, bool]]:
        """Gets agent's location."""
        metadata = self.controller.last_event.metadata
        location = {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"],
            "standing": metadata.get("isStanding", metadata["agent"].get("isStanding")),
        }
        return location

    @staticmethod
    def _agent_location_to_tuple(p: Dict[str, float]) -> Tuple[float, float]:
        return round(p["x"], 2), round(p["z"], 2)

    def _snap_agent_to_initially_reachable(self, verbose=True):
        agent_location = self.get_agent_location()

        end_location_tuple = self._agent_location_to_tuple(agent_location)
        if end_location_tuple in self.initially_reachable_points_set:
            return

        agent_x = agent_location["x"]
        agent_z = agent_location["z"]

        closest_reachable_points = list(self.initially_reachable_points_set)
        closest_reachable_points = sorted(
            closest_reachable_points,
            key=lambda xz: abs(xz[0] - agent_x) + abs(xz[1] - agent_z),
        )

        # In rare cases end_location_tuple might be not considered to be in self.initially_reachable_points_set
        # even when it is, here we check for such cases.
        if (
            math.sqrt(
                (
                    (
                        np.array(closest_reachable_points[0])
                        - np.array(end_location_tuple)
                    )
                    ** 2
                ).sum()
            )
            < 1e-6
        ):
            return

        saved_last_action = self.last_action
        saved_last_action_success = self.last_action_success
        saved_last_action_return = self.last_action_return
        saved_error_message = self.last_event.metadata["errorMessage"]

        # Thor behaves weirdly when the agent gets off of the grid and you
        # try to teleport the agent back to the closest grid location. To
        # get around this we first teleport the agent to random location
        # and then back to where it should be.
        for point in self.initially_reachable_points:
            if abs(agent_x - point["x"]) > 0.1 or abs(agent_z - point["z"]) > 0.1:
                self.teleport_agent_to(
                    rotation=0,
                    horizon=30,
                    **point,
                    only_initially_reachable=False,
                    verbose=False,
                )
                if self.last_action_success:
                    break

        for p in closest_reachable_points:
            self.teleport_agent_to(
                **{**agent_location, "x": p[0], "z": p[1]},
                only_initially_reachable=False,
                verbose=False,
            )
            if self.last_action_success:
                break

        teleport_forced = False
        if not self.last_action_success:
            self.teleport_agent_to(
                **{
                    **agent_location,
                    "x": closest_reachable_points[0][0],
                    "z": closest_reachable_points[0][1],
                },
                force_action=True,
                only_initially_reachable=False,
                verbose=False,
            )
            teleport_forced = True

        self.last_action = saved_last_action
        self.last_action_success = saved_last_action_success
        self.last_action_return = saved_last_action_return
        self.last_event.metadata["errorMessage"] = saved_error_message
        new_agent_location = self.get_agent_location()
        if verbose:
            warnings.warn(
                (
                    "In {}, at location (x,z)=({},{}) which is not in the set "
                    "of initially reachable points;"
                    " attempting to correct this: agent teleported to (x,z)=({},{}).\n"
                    "Teleportation {} forced."
                ).format(
                    self.scene_name,
                    agent_x,
                    agent_z,
                    new_agent_location["x"],
                    new_agent_location["z"],
                    "was" if teleport_forced else "wasn't",
                )
            )

    def step(
        self, action_dict: Dict[str, Union[str, int, float]]
    ) -> ai2thor.server.Event:
        """Take a step in the ai2thor environment."""
        action = typing.cast(str, action_dict["action"])

        skip_render = "renderImage" in action_dict and not action_dict["renderImage"]
        last_frame: Optional[np.ndarray] = None
        if skip_render:
            last_frame = self.current_frame

        if self.simplify_physics:
            action_dict["simplifyOPhysics"] = True

        self.last_frame_cache = self.current_frame.copy()
        self.last_depth_cache = self.current_depth.copy()

        if "moveMagnitude" in action_dict.keys() and "objectId" not in action_dict.keys():
            if "degrees" in action_dict.keys():
                drift_action_dict = {"action": "RotateRight",
                               "degrees": action_dict["degrees"]}
                _ = self.controller.step(drift_action_dict)
            action_dict = {"action": action,
                           "moveMagnitude": action_dict["moveMagnitude"]}
            start_location = self.get_agent_location()
            sr = self.controller.step(action_dict)

            if self.restrict_to_initially_reachable_points:
                end_location_tuple = self._agent_location_to_tuple(
                    self.get_agent_location()
                )
                if end_location_tuple not in self.initially_reachable_points_set:
                    self.teleport_agent_to(**start_location, force_action=True)  # type: ignore
                    self.last_action = action
                    self.last_action_success = False
                    self.last_event.metadata[
                        "errorMessage"
                    ] = "Moved to location outside of initially reachable points."
        elif "Move" in action and "Hand" not in action:  # type: ignore
            action_dict = {
                **action_dict,
                "moveMagnitude": self._move_mag,
            }  # type: ignore
            start_location = self.get_agent_location()
            sr = self.controller.step(action_dict)

            if self.restrict_to_initially_reachable_points:
                end_location_tuple = self._agent_location_to_tuple(
                    self.get_agent_location()
                )
                if end_location_tuple not in self.initially_reachable_points_set:
                    self.teleport_agent_to(**start_location, force_action=True)  # type: ignore
                    self.last_action = action
                    self.last_action_success = False
                    self.last_event.metadata[
                        "errorMessage"
                    ] = "Moved to location outside of initially reachable points."
        elif "RandomizeHideSeekObjects" in action:
            last_position = self.get_agent_location()
            self.controller.step(action_dict)
            metadata = self.last_event.metadata
            if self.position_dist(last_position, self.get_agent_location()) > 0.001:
                self.teleport_agent_to(**last_position, force_action=True)  # type: ignore
                warnings.warn(
                    "In scene {}, after randomization of hide and seek objects, agent moved.".format(
                        self.scene_name
                    )
                )

            sr = self.controller.step({"action": "GetReachablePositions"})
            self._initially_reachable_points = self.controller.last_event.metadata[
                "reachablePositions"
            ]
            self._initially_reachable_points_set = None
            self.last_action = action
            self.last_action_success = metadata["lastActionSuccess"]
            self.controller.last_event.metadata["reachablePositions"] = []
        elif "RotateUniverse" in action:
            sr = self.controller.step(action_dict)
            metadata = self.last_event.metadata

            if metadata["lastActionSuccess"]:
                sr = self.controller.step({"action": "GetReachablePositions"})
                self._initially_reachable_points = self.controller.last_event.metadata[
                    "reachablePositions"
                ]
                self._initially_reachable_points_set = None
                self.last_action = action
                self.last_action_success = metadata["lastActionSuccess"]
                self.controller.last_event.metadata["reachablePositions"] = []
        # elif "teleport" in action:
        #     obj = action_dict['object']
        #     target = action_dict['target']
        #     interactable_positions = self._interactable_positions_cache.get(
        #         scene_name=self.scene_name, obj=obj, controller=self.controller,
        #     )
        #     obj_pos = obj['position']
        #     # 保留xyz坐标
        #     interactable_positions = [{'x': pos['x'], 'y': pos['y'], 'z': pos['z']} for pos in interactable_positions if pos['standing'] is True and pos['horizon'] == 30]
        #     # 去除目标侧边的点
        #     interactable_positions = [pos for pos in interactable_positions if (pos['x']-obj_pos['x'])*(target['x']-obj_pos['x'])<0 and (pos['z']-obj_pos['z'])*(target['z']-obj_pos['z'])<0]
        #     dis_to_target = [self.position_dist(dict(x=pos['x'], z=pos['z'], y=0), target, ignore_y=True) for pos in interactable_positions]
        #     if len(dis_to_target) > 0:
        #         position = interactable_positions[dis_to_target.index(max(dis_to_target))]
        #         rot = (self.azimuthAngle(obj_pos['x'], obj_pos['z'], position['x'], position['z']) + 180) % 360
        #         # print(rot)
        #         sr = self.controller.step({'action': "Teleport",
        #                                    'position': position,
        #                                    'rotation': dict(x=0, y=rot, z=0),
        #                                    'horizon': 30})
        #         # print(position, obj_pos, target)
            else:
                # print("no pose")
                sr = self.controller.step(action='Done')
        else:
            sr = self.controller.step(action_dict)


        if self.restrict_to_initially_reachable_points:
            self._snap_agent_to_initially_reachable()

        if skip_render:
            assert last_frame is not None
            self.last_event.frame = last_frame

        return sr

    @staticmethod
    def position_dist(
        p0: Mapping[str, Any], p1: Mapping[str, Any], ignore_y: bool = False
    ) -> float:
        """Distance between two points of the form {"x": x, "y":y, "z":z"}."""
        return math.sqrt(
            (p0["x"] - p1["x"]) ** 2
            + (0 if ignore_y else (p0["y"] - p1["y"]) ** 2)
            + (p0["z"] - p1["z"]) ** 2
        )

    @staticmethod
    def rotation_dist(a: Dict[str, float], b: Dict[str, float]):
        """Distance between rotations."""

        def deg_dist(d0: float, d1: float):
            dist = (d0 - d1) % 360
            return min(dist, 360 - dist)

        return sum(deg_dist(a[k], b[k]) for k in ["x", "y", "z"])

    def closest_object_with_properties(
        self, properties: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find the object closest to the agent that has the given
        properties."""
        agent_pos = self.controller.last_event.metadata["agent"]["position"]
        min_dist = float("inf")
        closest = None
        for o in self.all_objects():
            satisfies_all = True
            for k, v in properties.items():
                if o[k] != v:
                    satisfies_all = False
                    break
            if satisfies_all:
                d = self.position_dist(agent_pos, o["position"])
                if d < min_dist:
                    min_dist = d
                    closest = o
        return closest

    def closest_visible_object_of_type(
        self, object_type: str
    ) -> Optional[Dict[str, Any]]:
        """Find the object closest to the agent that is visible and has the
        given type."""
        properties = {"visible": True, "objectType": object_type}
        return self.closest_object_with_properties(properties)

    def closest_object_of_type(self, object_type: str) -> Optional[Dict[str, Any]]:
        """Find the object closest to the agent that has the given type."""
        properties = {"objectType": object_type}
        return self.closest_object_with_properties(properties)

    def closest_reachable_point_to_position(
        self, position: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
        """Of all reachable positions, find the one that is closest to the
        given location."""
        target = np.array([position["x"], position["z"]])
        min_dist = float("inf")
        closest_point = None
        for pt in self.initially_reachable_points:
            dist = np.linalg.norm(target - np.array([pt["x"], pt["z"]]))
            if dist < min_dist:
                closest_point = pt
                min_dist = dist
                if min_dist < 1e-3:
                    break
        assert closest_point is not None
        return closest_point, min_dist

    @staticmethod
    def _angle_from_to(a_from: float, a_to: float) -> float:
        a_from = a_from % 360
        a_to = a_to % 360
        min_rot = min(a_from, a_to)
        max_rot = max(a_from, a_to)
        rot_across_0 = (360 - max_rot) + min_rot
        rot_not_across_0 = max_rot - min_rot
        rot_err = min(rot_across_0, rot_not_across_0)
        if rot_across_0 == rot_err:
            rot_err *= -1 if a_to > a_from else 1
        else:
            rot_err *= 1 if a_to > a_from else -1
        return rot_err

    def agent_xz_to_scene_xz(self, agent_xz: Dict[str, float]) -> Dict[str, float]:
        agent_pos = self.get_agent_location()

        x_rel_agent = agent_xz["x"]
        z_rel_agent = agent_xz["z"]
        scene_x = agent_pos["x"]
        scene_z = agent_pos["z"]
        rotation = agent_pos["rotation"]
        if abs(rotation) < 1e-5:
            scene_x += x_rel_agent
            scene_z += z_rel_agent
        elif abs(rotation - 90) < 1e-5:
            scene_x += z_rel_agent
            scene_z += -x_rel_agent
        elif abs(rotation - 180) < 1e-5:
            scene_x += -x_rel_agent
            scene_z += -z_rel_agent
        elif abs(rotation - 270) < 1e-5:
            scene_x += -z_rel_agent
            scene_z += x_rel_agent
        else:
            raise Exception("Rotation must be one of 0, 90, 180, or 270.")

        return {"x": scene_x, "z": scene_z}

    def scene_xz_to_agent_xz(self, scene_xz: Dict[str, float]) -> Dict[str, float]:
        agent_pos = self.get_agent_location()
        x_err = scene_xz["x"] - agent_pos["x"]
        z_err = scene_xz["z"] - agent_pos["z"]

        rotation = agent_pos["rotation"]
        if abs(rotation) < 1e-5:
            agent_x = x_err
            agent_z = z_err
        elif abs(rotation - 90) < 1e-5:
            agent_x = -z_err
            agent_z = x_err
        elif abs(rotation - 180) < 1e-5:
            agent_x = -x_err
            agent_z = -z_err
        elif abs(rotation - 270) < 1e-5:
            agent_x = z_err
            agent_z = -x_err
        else:
            raise Exception("Rotation must be one of 0, 90, 180, or 270.")

        return {"x": agent_x, "z": agent_z}

    def all_objects(self) -> List[Dict[str, Any]]:
        """Return all object metadata."""
        return self.controller.last_event.metadata["objects"]

    def all_objects_with_properties(
        self, properties: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find all objects with the given properties."""
        objects = []
        for o in self.all_objects():
            satisfies_all = True
            for k, v in properties.items():
                if o[k] != v:
                    satisfies_all = False
                    break
            if satisfies_all:
                objects.append(o)
        return objects

    def visible_objects(self) -> List[Dict[str, Any]]:
        """Return all visible objects."""
        return self.all_objects_with_properties({"visible": True})

    def get_object_by_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        for o in self.last_event.metadata["objects"]:
            if o["objectId"] == object_id:
                return o
        return None

    ###
    # Following is used for computing shortest paths between states
    ###
    _CACHED_GRAPHS: Dict[str, nx.DiGraph] = {}

    GRAPH_ACTIONS_SET = {"LookUp", "LookDown", "RotateLeft", "RotateRight", "MoveAhead"}

    def reachable_points_with_rotations_and_horizons(self):
        self.controller.step({"action": "GetReachablePositions"})
        assert self.last_action_success

        points_slim = self.last_event.metadata["actionReturn"]

        points = []
        for r in [0, 90, 180, 270]:
            for horizon in [-30, 0, 30, 60]:
                for p in points_slim:
                    p = copy.copy(p)
                    p["rotation"] = r
                    p["horizon"] = horizon
                    points.append(p)
        return points

    @staticmethod
    def location_for_key(key, y_value=0.0):
        x, z, rot, hor = key
        loc = dict(x=x, y=y_value, z=z, rotation=rot, horizon=hor)
        return loc

    @staticmethod
    def get_key(input_dict: Dict[str, Any]) -> Tuple[float, float, int, int]:
        if "x" in input_dict:
            x = input_dict["x"]
            z = input_dict["z"]
            rot = input_dict["rotation"]
            hor = input_dict["horizon"]
        else:
            x = input_dict["position"]["x"]
            z = input_dict["position"]["z"]
            rot = input_dict["rotation"]["y"]
            hor = input_dict["cameraHorizon"]

        return (
            round(x, 2),
            round(z, 2),
            round_to_factor(rot, 90) % 360,
            round_to_factor(hor, 30) % 360,
        )

    def update_graph_with_failed_action(self, failed_action: str):
        if (
            self.scene_name not in self._CACHED_GRAPHS
            or failed_action not in self.GRAPH_ACTIONS_SET
        ):
            return

        source_key = self.get_key(self.last_event.metadata["agent"])
        edge_dict = self.graph[source_key]
        to_remove_key = None
        for target_key in self.graph[source_key]:
            if edge_dict[target_key]["action"] == failed_action:
                to_remove_key = target_key
                break
        if to_remove_key is not None:
            self.graph.remove_edge(source_key, to_remove_key)

    def _add_from_to_edge(
        self,
        g: nx.DiGraph,
        s: Tuple[float, float, int, int],
        t: Tuple[float, float, int, int],
    ):
        def ae(x, y):
            return abs(x - y) < 0.001

        s_x, s_z, s_rot, s_hor = s
        t_x, t_z, t_rot, t_hor = t

        dist = round(math.sqrt((s_x - t_x) ** 2 + (s_z - t_z) ** 2), 2)
        angle_dist = (round_to_factor(t_rot - s_rot, 90) % 360) // 90
        horz_dist = (round_to_factor(t_hor - s_hor, 30) % 360) // 30

        # If source and target differ by more than one action, continue
        if sum(x != 0 for x in [dist, angle_dist, horz_dist]) != 1:
            return

        grid_size = self._grid_size
        action = None
        if angle_dist != 0:
            if angle_dist == 1:
                action = "RotateRight"
            elif angle_dist == 3:
                action = "RotateLeft"

        elif horz_dist != 0:
            if horz_dist == 11:
                action = "LookUp"
            elif horz_dist == 1:
                action = "LookDown"
        elif ae(dist, grid_size):
            if (
                (s_rot == 0 and ae(t_z - s_z, grid_size))
                or (s_rot == 90 and ae(t_x - s_x, grid_size))
                or (s_rot == 180 and ae(t_z - s_z, -grid_size))
                or (s_rot == 270 and ae(t_x - s_x, -grid_size))
            ):
                g.add_edge(s, t, action="MoveAhead")

        if action is not None:
            g.add_edge(s, t, action=action)

    @functools.lru_cache(1)
    def possible_neighbor_offsets(self) -> Tuple[Tuple[float, float, int, int], ...]:
        grid_size = round(self._grid_size, 2)
        offsets = []
        for rot_diff in [-90, 0, 90]:
            for horz_diff in [-30, 0, 30, 60]:
                for x_diff in [-grid_size, 0, grid_size]:
                    for z_diff in [-grid_size, 0, grid_size]:
                        if (rot_diff != 0) + (horz_diff != 0) + (x_diff != 0) + (
                            z_diff != 0
                        ) == 1:
                            offsets.append((x_diff, z_diff, rot_diff, horz_diff))
        return tuple(offsets)

    def _add_node_to_graph(self, graph: nx.DiGraph, s: Tuple[float, float, int, int]):
        if s in graph:
            return

        existing_nodes = set(graph.nodes())
        graph.add_node(s)

        for o in self.possible_neighbor_offsets():
            t = (s[0] + o[0], s[1] + o[1], s[2] + o[2], s[3] + o[3])
            if t in existing_nodes:
                self._add_from_to_edge(graph, s, t)
                self._add_from_to_edge(graph, t, s)

    @property
    def graph(self):
        if self.scene_name not in self._CACHED_GRAPHS:
            g = nx.DiGraph()
            points = self.reachable_points_with_rotations_and_horizons()
            for p in points:
                self._add_node_to_graph(g, self.get_key(p))

            self._CACHED_GRAPHS[self.scene_name] = g
        return self._CACHED_GRAPHS[self.scene_name]

    @graph.setter
    def graph(self, g):
        self._CACHED_GRAPHS[self.scene_name] = g

    def _check_contains_key(self, key: Tuple[float, float, int, int], add_if_not=True):
        if key not in self.graph:
            warnings.warn(
                "{} was not in the graph for scene {}.".format(key, self.scene_name)
            )
            if add_if_not:
                self._add_node_to_graph(self.graph, key)

    def shortest_state_path(self, source_state_key, goal_state_key):
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        # noinspection PyBroadException
        try:
            path = nx.shortest_path(self.graph, source_state_key, goal_state_key)
            return path
        except Exception as _:
            return None

    def action_transitioning_between_keys(self, s, t):
        self._check_contains_key(s)
        self._check_contains_key(t)
        if self.graph.has_edge(s, t):
            return self.graph.get_edge_data(s, t)["action"]
        else:
            return None

    def shortest_path_next_state(self, source_state_key, goal_state_key):
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        if source_state_key == goal_state_key:
            raise RuntimeError("called next state on the same source and goal state")
        state_path = self.shortest_state_path(source_state_key, goal_state_key)
        return state_path[1]

    def shortest_path_next_action(self, source_state_key, goal_state_key):
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)

        next_state_key = self.shortest_path_next_state(source_state_key, goal_state_key)
        return self.graph.get_edge_data(source_state_key, next_state_key)["action"]

    def shortest_path_length(self, source_state_key, goal_state_key):
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        try:
            return nx.shortest_path_length(self.graph, source_state_key, goal_state_key)
        except nx.NetworkXNoPath as _:
            return float("inf")

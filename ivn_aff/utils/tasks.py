from rearrange_on_proc.arguments import args
from explore import Explore
from visdom import Visdom
from visualization import Animation, Visualize_Obj
from rearrange import Rearrange
from rearrange_on_proc.datagen.datagen_constants import OBJECT_TYPES_TO_NOT_MOVE
from rearrange_on_proc.utils.utils import round_to_factor, pose_difference_energy
from allenact.utils.misc_utils import NumpyJSONEncoder
# from utils import round_to_factor,_obj_list_to_obj_name_to_pose_dict

import time
import numpy as np
import json
import pickle
import os
import compress_pickle

class RearrangementTask():
    def __init__(self, controller, task, task_areaRange, all_dataset, process_id, task_index, solution_config, online_controller = None) -> None:
        self.controller = controller        #offline or online
        self.online_controller = online_controller #online (only used after offline)
        self.curr_task = task 
        self.curr_task_areaRange = task_areaRange
        self.dataset = all_dataset
        self.process_id = process_id
        self.task_index = task_index
        self.solution_config = solution_config
        # self.solution_str = f"{args.walkthrough_search}|{args.unshuffle_search}|{args.unshuffle_match}"
        self.solution_str = f"{args.walkthrough_search}|{args.unshuffle_search}|{args.seg_utils}"

    def set_walkthrough_or_unshuffle_env(self, task_index, stage, curr_scene_unique_id):
        # First: Reset the env (只在walkthrough: lwj?)
        house_data = self.dataset[self.curr_scene_split][self.curr_scene_id]
        room_num = len(house_data['rooms'])
        if stage == 'walkthrough':
            if args.use_offline_ai2thor:
                self.controller.reset(scene_name = curr_scene_unique_id, stage_name = stage)
                self.online_controller.reset(scene = house_data)
            else:
                self.controller.reset(scene = house_data)
            print(f'====For Process {self.process_id}-task {task_index}: house {self.curr_scene_unique_id} with {room_num} rooms====')
            assert self.controller.last_event.metadata['lastActionSuccess'] == True, f"Reset [Env] Fails!!!"
            if self.curr_task_areaRange == '<10':
                args.map_size = 5 * 2
                args.step_max = 50
                args.rearrange_step_max = 30
            elif self.curr_task_areaRange == '10-60':
                args.map_size = 8 * 2    #10 * 2
                # args.step_max = 20
                args.step_max = 200
                args.rearrange_step_max = 80
            elif self.curr_task_areaRange == '60-150':
                args.map_size = 14 * 2    #15 * 2
                args.step_max = 300
                args.rearrange_step_max = 100
            elif self.curr_task_areaRange == '150-300':
                args.map_size = 20 * 2    #20 * 2
                args.step_max = 500
                args.rearrange_step_max = 200
            elif self.curr_task_areaRange == '>300':
                args.map_size = 30 * 2
                args.step_max = 800
                args.rearrange_step_max = 300
        elif stage == 'unshuffle':
            if args.use_offline_ai2thor:
                self.controller.reset(scene_name = curr_scene_unique_id, stage_name = stage)
                # self.online_controller.reset(scene = house_data)  
        
        print(f"---:p{self.process_id}:{self.curr_task['unique_id']} ---「{stage}」---")
        # Second: Set agent position
        pos = self.curr_task['agent_position']
        rotation_y = self.curr_task['agent_rotation']
        rotation_y = round_to_factor(rotation_y, 90)
        rot = {"x": 0, "y": rotation_y, "z": 0}
        # lwj:原TIDEE将所有旋转进行了align，0，90，180，270
        if args.use_offline_ai2thor:
            self.controller.step_teleportfull(
                **pos,
                rotation = rotation_y,
                horizon = 0.0
            )
            self.online_controller.step(
                "TeleportFull",
                **pos,
                rotation=rot,
                horizon=0.0,  #lyy注意检查一下这里
                standing=True,
                forceAction=True,
            )
            assert self.online_controller.last_event.metadata['lastActionSuccess'] == True
        else:
            self.controller.step(
                "TeleportFull",
                **pos,
                rotation=rot,
                horizon=0.0,  #lyy注意检查一下这里
                standing=True,
                forceAction=True,
            )
        assert self.controller.last_event.metadata['lastActionSuccess'] == True, f"Reset [Agent] Fails!!!"

        # the offline_ai2thor has already set the environment, only prepare the online_controller for the Rearrange Stage
        if args.use_offline_ai2thor:
            # Third: Set object position
            if stage == 'walkthrough':
                curr_stage_pickupable_objects_poses = self.curr_task['target_poses']
            elif stage == 'unshuffle':
                curr_stage_pickupable_objects_poses= self.curr_task['starting_poses']
            considered_pickupable_object_names =  [obj['name'] for obj in curr_stage_pickupable_objects_poses]
            kept_object_poses = []
            for l_obj in self.online_controller.last_event.metadata["objects"]:
                if not l_obj['pickupable'] and not l_obj['moveable']:
                    continue
                if l_obj['pickupable'] and l_obj['name'] not in considered_pickupable_object_names:
                    continue
                if l_obj['name'] not in considered_pickupable_object_names and l_obj['objectType'] not in OBJECT_TYPES_TO_NOT_MOVE:
                    kept_object_poses.append({
                        "objectId": l_obj['objectId'],
                        "objectName": l_obj['name'],
                        "position": l_obj['position'],
                        "rotation": l_obj['rotation'],
                    })

            self.online_controller.step(
                "SetObjectPoses",
                objectPoses=curr_stage_pickupable_objects_poses + kept_object_poses,
                placeStationary=True,
                enablePhysicsJitter=True,
                forceRigidbodySleep=True,
                skipMoveable=True,
            )
            assert self.online_controller.last_event.metadata['lastActionSuccess'] == True, f"Set objects positions Fails!!! {self.online_controller.last_event.metadata['errorMessage']}"

            # Fourth: Open object
            curr_stage_object_openness_str = 'target_openness' if stage == 'walkthrough' else 'start_openness'
            for obj in self.curr_task['openable_data']:
                # id is re-found due to possible floating point errors
                current_obj_info = next(
                    l_obj for l_obj in self.online_controller.last_event.metadata["objects"]
                    if l_obj["objectId"] == obj["objectId"]
                )
                self.online_controller.step(
                    action="OpenObject",
                    objectId=current_obj_info["objectId"],
                    openness=obj[curr_stage_object_openness_str],
                    forceAction=True,
                )
                assert self.online_controller.last_event.metadata['lastActionSuccess'] == True, f"Open object Fails!!! {self.online_controller.last_event.metadata['errorMessage']}"

            self.online_controller.step('Pass')
        else:
            # Third: Set object position
            if stage == 'walkthrough':
                curr_stage_pickupable_objects_poses = self.curr_task['target_poses']
            elif stage == 'unshuffle':
                curr_stage_pickupable_objects_poses= self.curr_task['starting_poses']
            considered_pickupable_object_names =  [obj['name'] for obj in curr_stage_pickupable_objects_poses]
            kept_object_poses = []
            for l_obj in self.controller.last_event.metadata["objects"]:
                if not l_obj['pickupable'] and not l_obj['moveable']:
                    continue
                if l_obj['pickupable'] and l_obj['name'] not in considered_pickupable_object_names:
                    continue
                if l_obj['name'] not in considered_pickupable_object_names and l_obj['objectType'] not in OBJECT_TYPES_TO_NOT_MOVE:
                    kept_object_poses.append({
                        "objectId": l_obj['objectId'],
                        "objectName": l_obj['name'],
                        "position": l_obj['position'],
                        "rotation": l_obj['rotation'],
                    })

            self.controller.step(
                "SetObjectPoses",
                objectPoses=curr_stage_pickupable_objects_poses + kept_object_poses,
                placeStationary=True,
                enablePhysicsJitter=True,
                forceRigidbodySleep=True,
                skipMoveable=True,
            )
            assert self.controller.last_event.metadata['lastActionSuccess'] == True, f"Set objects positions Fails!!! {self.controller.last_event.metadata['errorMessage']}"

            # Fourth: Open object
            curr_stage_object_openness_str = 'target_openness' if stage == 'walkthrough' else 'start_openness'
            for obj in self.curr_task['openable_data']:
                # id is re-found due to possible floating point errors
                current_obj_info = next(
                    l_obj for l_obj in self.controller.last_event.metadata["objects"]
                    if l_obj["objectId"] == obj["objectId"]
                )
                self.controller.step(
                    action="OpenObject",
                    objectId=current_obj_info["objectId"],
                    openness=obj[curr_stage_object_openness_str],
                    forceAction=True,
                )
                assert self.controller.last_event.metadata['lastActionSuccess'] == True, f"Open object Fails!!! {self.controller.last_event.metadata['errorMessage']}"

            self.controller.step('Pass')
    
    def main(self, ):
        start_time = time.perf_counter()
        print(f"========Process {self.process_id}: {self.curr_task['unique_id']} task=========")

        split, scene_id, reuse_id = self.curr_task['unique_id'].split('_')
        if split == 'debug':
            split = 'train'
        scene_id = int(scene_id)
        reuse_id = int(reuse_id)
        self.curr_scene_split = split
        self.curr_scene_id = scene_id
        self.curr_scene_reuse_index = reuse_id
        self.curr_scene_unique_id = self.curr_task['unique_id']
        self.curr_scene_roomSpec = self.curr_task['roomSpec']

        # 用于评估
        self.curr_metrics = {
            'split': self.curr_scene_split,
            'scene': self.curr_scene_id,
            'reuse_index': self.curr_scene_reuse_index,
            'unique_id': self.curr_scene_unique_id,
        }
        
        #判断是否需要可视化
        if args.create_movie:
            vis = Animation(args.W,args.H)
        else:
            vis = None
        if args.generate_rearrangement_images:
            vis_rearrange = Visualize_Obj()

        # visdom = Visdom(env = f'{self.curr_scene_split}-{self.curr_scene_id}-{self.curr_scene_roomSpec}')
        visdom = None
        # visdom = Visdom(env = 'debug')

        # 1.0. 设置walkthrough的环境controller
        self.curr_stage = 'walkthrough'
        self.set_walkthrough_or_unshuffle_env(self.task_index, self.curr_stage, self.curr_scene_unique_id)
        # 1.1. Walkthrough阶段机器人探索房间
        roomNum = None
        if args.roomNum == 'all':
            roomNum = 'all'
        else:
            roomNum = [str(x)+'rooms' for x in args.roomNum.split('|')]
        if args.test_mode == "only_walkthrough":
            args.step_max = args.only_walkthrough_mode_max_steps
            args.movie_dir = f"./movies_{str(roomNum)}_{args.seg_utils}{args.test_mode}|{args.walkthrough_search}|{args.only_walkthrough_mode_max_steps}"
        elif args.test_mode == 'only_walkthrough_steps':
            args.step_max = 1000
            args.movie_dir = f"./movies_{str(roomNum)}_{args.seg_utils}{args.test_mode}|{args.walkthrough_search}|{args.only_walkthrough_mode_max_steps}"
        else:
            f"./movies_{str(roomNum)}_{args.seg_utils}{args.test_mode}|{args.walkthrough_search}|{args.unshuffle_search}|{args.unshuffle_match}|{args.unshuffle_reorder}"
        walkthrough_explorer = Explore(controller=self.controller, vis=vis, visdom=visdom, curr_stage = self.curr_stage, map_size=args.map_size, step_max = args.step_max, process_id = self.process_id,solution_config=self.solution_config)
        if not args.load_walkthrough_explorer:
            walkthrough_objs_id_to_pose_init = walkthrough_explorer.get_current_objs_id_to_pose(stage = self.curr_stage)
            walkthrough_explorer.explore_env() #with: action, action_success and reward, ep_length, and other explore metrics.
            walkthrough_explore_metrics = walkthrough_explorer.get_metrics()
            print(f'Walkthrough explore takes {walkthrough_explorer.step_from_stage_start} steps!')
            # obj_name_to_walkthrough_start_pose = _obj_list_to_obj_name_to_pose_dict(self.controller.last_event.metadata["objects"])
            if args.generate_rearrangement_images:
                vis_rearrange.save_view_distance_vis(walkthrough_explorer, self.curr_scene_unique_id, self.curr_stage)
                vis_rearrange.save_semantic_map(walkthrough_explorer, self.curr_scene_unique_id, self.curr_stage)                
                vis_rearrange.get_walkthrough_images(walkthrough_explorer, self.online_controller)
            #保存walkthrough阶段的信息
            if args.store_walkthrough_explorer:
                #地图信息
                compress_pickle.dump(
                    obj=walkthrough_explorer.mapper,
                    path=os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_walkthrough_mapper.pkl.gz"),
                    pickler_kwargs={"protocol": 4,},  
                )
                #obj_tracker
                
                compress_pickle.dump(
                    obj=walkthrough_explorer.object_tracker,
                    path=os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_walkthrough_object_tracker.pkl.gz"),
                    pickler_kwargs={"protocol": 4,},  
                )
                #可视化信息
                if args.generate_rearrangement_images:
                    compress_pickle.dump(
                        obj=vis_rearrange,
                        path=os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_walkthrough_vis_rearrange.pkl.gz"),
                        pickler_kwargs={"protocol": 4,},  
                    )
                if args.create_movie:
                    compress_pickle.dump(
                        obj=vis.image_plots,
                        path=os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_walkthrough_vis.pkl.gz"),
                        pickler_kwargs={"protocol": 4,},  
                    )
                #其他信息
                other_info = {}
                other_info["walkthrough_objs_id_to_pose_init"] = walkthrough_objs_id_to_pose_init
                other_info["walkthrough_explore_metrics"] = walkthrough_explore_metrics
                with open(args.store_explorer_path+"/"+self.curr_scene_unique_id+'_'+self.solution_str+'_'+"walkthrough_other_info.json", "w") as f:
                    json.dump(other_info,f)
                print("store walkthrough explorer success!")
                
        else:
            walkthrough_explorer.mapper = compress_pickle.load(os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_walkthrough_mapper.pkl.gz"))

            #obj_tracker   
            
            walkthrough_explorer.object_tracker = compress_pickle.load(os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_walkthrough_object_tracker.pkl.gz"))

            #可视化信息
            if args.generate_rearrangement_images:
                vis_rearrange = compress_pickle.load(os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_walkthrough_vis_rearrange.pkl.gz"))

            if args.create_movie:
                vis.image_plots = compress_pickle.load(os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_walkthrough_vis.pkl.gz"))

            #其他信息
            with open(args.store_explorer_path+"/"+self.curr_scene_unique_id+'_'+self.solution_str+'_'+"walkthrough_other_info.json", "r") as f:
                other_info = json.load(f)
            walkthrough_objs_id_to_pose_init = other_info["walkthrough_objs_id_to_pose_init"]
            walkthrough_explore_metrics = other_info["walkthrough_explore_metrics"]
            print("load walkthrough explorer success!")

            if args.generate_rearrangement_images:
                vis_rearrange.save_view_distance_vis(walkthrough_explorer, self.curr_scene_unique_id, self.curr_stage)
                vis_rearrange.save_semantic_map(walkthrough_explorer, self.curr_scene_unique_id, self.curr_stage)                
 
        if args.test_mode == "only_walkthrough" or args.test_mode == 'only_walkthrough_steps':
            self.curr_metrics = {
            **self.curr_metrics,
            'ep_length': walkthrough_explore_metrics['walkthrough/ep_length'],
            **walkthrough_explore_metrics,
            }
            return (self.curr_metrics['unique_id'], self.curr_metrics,0,0)
            

        # 2.0. 设置unshuffle的环境controller
        self.curr_stage = 'unshuffle'
        self.set_walkthrough_or_unshuffle_env(self.task_index, self.curr_stage, self.curr_scene_unique_id)
        # 2.1. Unshuffle阶段机器人探索房间
        unshuffle_explorer = Explore(controller=self.controller,vis=vis, visdom=visdom,  curr_stage = self.curr_stage, map_size = args.map_size, step_max = args.step_max, process_id = self.process_id, solution_config=self.solution_config,walkthrough_objs_id_to_pose=walkthrough_objs_id_to_pose_init)
        if not args.load_unshuffle_explorer:
            self.unshuffle_objs_id_to_pose = unshuffle_explorer.get_current_objs_id_to_pose(stage = self.curr_stage)
            self.walkthrough_objs_id_to_pose = unshuffle_explorer.walkthrough_objs_id_to_pose
            unshuffle_explorer.explore_env()
            print(f'Unshuffle explore takes {unshuffle_explorer.step_from_stage_start} steps')
            if args.generate_rearrangement_images:
                vis_rearrange.save_view_distance_vis(unshuffle_explorer, self.curr_scene_unique_id, self.curr_stage)
                vis_rearrange.save_semantic_map(unshuffle_explorer, self.curr_scene_unique_id, self.curr_stage)                
                vis_rearrange.get_unshuffle_images(unshuffle_explorer, self.online_controller)

            #保存unshuffle阶段的信息
            if args.store_unshuffle_explorer:
                #地图信息
                compress_pickle.dump(
                    obj=unshuffle_explorer.mapper,
                    path=os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_unshuffle_mapper.pkl.gz"),
                    pickler_kwargs={"protocol": 4,},  
                )
                #obj_tracker
                compress_pickle.dump(
                    obj=unshuffle_explorer.object_tracker,
                    path=os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_unshuffle_object_tracker.pkl.gz"),
                    pickler_kwargs={"protocol": 4,},  
                )
                #可视化信息
                if args.generate_rearrangement_images:
                    compress_pickle.dump(
                        obj=vis_rearrange,
                        path=os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_unshuffle_vis_rearrange.pkl.gz"),
                        pickler_kwargs={"protocol": 4,},  
                    )
                if args.create_movie:
                    compress_pickle.dump(
                        obj=vis.image_plots,
                        path=os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_unshuffle_vis.pkl.gz"),
                        pickler_kwargs={"protocol": 4,},  
                    )
                #其他信息
                other_info = {}
                other_info["unshuffle_objs_id_to_pose"] = self.unshuffle_objs_id_to_pose
                other_info["walkthrough_objs_id_to_pose"] = self.walkthrough_objs_id_to_pose
                other_info["unshuffle_afterExplore_metrics"] = unshuffle_explorer.get_metrics()
                other_info["step_from_stage_start"] = unshuffle_explorer.step_from_stage_start
                other_info["simulate_last_position"] = unshuffle_explorer.controller.last_event.metadata["agent"]["position"]
                other_info["simulate_last_rotation"] = unshuffle_explorer.controller.last_event.metadata["agent"]["rotation"]
                other_info["simulate_last_horizon"] = unshuffle_explorer.controller.last_event.metadata["agent"]["cameraHorizon"]
                other_info["last_position"] = unshuffle_explorer.position
                other_info["last_rotation"] = unshuffle_explorer.rotation
                other_info["last_head_tilt"] = unshuffle_explorer.head_tilt
                other_info["seed"] = unshuffle_explorer.seed
                with open(args.store_explorer_path+"/"+self.curr_scene_unique_id+"_"+self.solution_str+"_"+"unshuffle_other_info.json", "w") as f:
                    json.dump(other_info,f)
                print("store unshuffle explorer success!")
        else:
            unshuffle_explorer.mapper = compress_pickle.load(os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_unshuffle_mapper.pkl.gz"))
      
            #obj_tracker
            unshuffle_explorer.object_tracker = compress_pickle.load(os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_unshuffle_object_tracker.pkl.gz"))

            #可视化信息
            if args.generate_rearrangement_images:
                vis_rearrange = compress_pickle.load(os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_unshuffle_vis_rearrange.pkl.gz"))

            if args.create_movie:
                vis.image_plots = compress_pickle.load(os.path.join(args.store_explorer_path, f"{self.curr_scene_unique_id}_{self.solution_str}_unshuffle_vis.pkl.gz"))
                unshuffle_explorer.vis = vis

            #其他信息
            with open(args.store_explorer_path+"/"+self.curr_scene_unique_id+"_"+self.solution_str+"_"+"unshuffle_other_info.json", "r") as f:
                other_info = json.load(f)
            self.unshuffle_objs_id_to_pose = other_info["unshuffle_objs_id_to_pose"]
            self.walkthrough_objs_id_to_pose = other_info["walkthrough_objs_id_to_pose"]
            simulate_last_position = other_info["simulate_last_position"]
            simulate_last_rotation = other_info["simulate_last_rotation"] 
            simulate_last_horizon = other_info["simulate_last_horizon"]
            unshuffle_explorer.metrics = other_info['unshuffle_afterExplore_metrics']
            unshuffle_explorer.step_from_stage_start = other_info['step_from_stage_start']
            unshuffle_explorer.position = other_info["last_position"]
            unshuffle_explorer.rotation = other_info["last_rotation"]
            unshuffle_explorer.head_tilt = other_info["last_head_tilt"]
            unshuffle_explorer.seed = other_info["seed"]
            if args.use_offline_ai2thor:
                unshuffle_explorer.controller.step_teleportfull(
                    **simulate_last_position,
                    rotation = simulate_last_rotation['y'],
                    horizon = int(simulate_last_horizon)
                )
            else:
                unshuffle_explorer.controller.step(
                    action="TeleportFull",
                    position=simulate_last_position,
                    rotation=simulate_last_rotation,
                    horizon = int(simulate_last_horizon),
                    standing = True
                )
            assert unshuffle_explorer.controller.last_event.metadata["lastActionSuccess"] == True
            unshuffle_explorer.rgb = unshuffle_explorer.controller.last_event.frame
            unshuffle_explorer.depth = unshuffle_explorer.controller.last_event.depth_frame
            print("load unshuffle explorer success!")
            
            if args.generate_rearrangement_images:
                vis_rearrange.save_view_distance_vis(unshuffle_explorer, self.curr_scene_unique_id, self.curr_stage)
                vis_rearrange.save_semantic_map(unshuffle_explorer, self.curr_scene_unique_id, self.curr_stage)                

        # 如果walkthrough和unshuffle的探索都是基于离线的模拟器，则整理时要换成在线的模拟器
        if args.use_offline_ai2thor:
            simulate_last_position = unshuffle_explorer.controller.last_event.metadata["agent"]["position"]
            simulate_last_rotation = unshuffle_explorer.controller.last_event.metadata["agent"]["rotation"]
            simulate_last_horizon = unshuffle_explorer.controller.last_event.metadata["agent"]["cameraHorizon"]
            unshuffle_explorer.controller = self.online_controller  #重要！！
            unshuffle_explorer.segHelper.controller = self.online_controller #重要！！

            unshuffle_explorer.controller.step(
                action="TeleportFull",
                position=simulate_last_position,
                rotation=simulate_last_rotation,
                horizon = int(simulate_last_horizon),
                standing = True
            )
            assert unshuffle_explorer.controller.last_event.metadata["lastActionSuccess"] == True
            unshuffle_explorer.rgb = unshuffle_explorer.controller.last_event.frame
            unshuffle_explorer.depth = unshuffle_explorer.controller.last_event.depth_frame

        explore_time = time.perf_counter()

        # 2.2. 整理收纳阶段
        rearranger = Rearrange(walkthrouth_explorer=walkthrough_explorer,unshuffle_explorer=unshuffle_explorer, process_id = self.process_id, task=self.curr_task, visdom=visdom, walkthrough_objs_id_to_pose= walkthrough_objs_id_to_pose_init,solution_config=self.solution_config)
        rearranger.rearange_objects() 
        self.curr_objs_id_to_pose = rearranger.unshuffle_explorer.get_current_objs_id_to_pose(stage='rearrange')

        #3.0. 计算metrics
        unshuffle_interact_metrics = rearranger.get_metrics() #with: actions, action_success, reward and ep_length
        unshuffle_obj_metrics, missed_obj, end_energies_dict = self._calculate_task_metrics() # with obj-relate metrics
        if args.generate_rearrangement_images:
            vis_rearrange.get_rearranged_images(unshuffle_explorer, end_energies_dict)

        self.curr_metrics = {
            **self.curr_metrics,
            'ep_length': walkthrough_explore_metrics['walkthrough/ep_length'] + unshuffle_interact_metrics['unshuffle/ep_length'],
            **walkthrough_explore_metrics,
            **unshuffle_interact_metrics,
            **unshuffle_obj_metrics,
        }
        
        # 生成探索过程的视频
        # if vis != None:
            # vis = unshuffle_explorer.vis
            # vis.render_movie(args.movie_dir, self.process_id, self.task_index, tag=f'{self.curr_scene_unique_id}_succ={self.curr_metrics["unshuffle/success"]}_pfs={self.curr_metrics["unshuffle/prop_fixed_strict"]}_missed={missed_obj}')
        # visdom.close()
        if args.generate_rearrangement_images:
            vis_rearrange.save_final_images(self.curr_scene_unique_id, tag=f'{self.curr_scene_unique_id}_succ={self.curr_metrics["unshuffle/success"]}_pfs={self.curr_metrics["unshuffle/prop_fixed_strict"]}_missed={missed_obj}')

        end_time = time.perf_counter()

        return (self.curr_metrics['unique_id'], self.curr_metrics, explore_time-start_time, end_time-start_time)


    def _calculate_task_metrics(self):
        start_energies_dict = pose_difference_energy(goal_poses=self.walkthrough_objs_id_to_pose, cur_poses=self.unshuffle_objs_id_to_pose)
        end_energies_dict = pose_difference_energy(goal_poses=self.walkthrough_objs_id_to_pose, cur_poses=self.curr_objs_id_to_pose)
        change_energies_dict = pose_difference_energy(goal_poses=self.unshuffle_objs_id_to_pose, cur_poses=self.curr_objs_id_to_pose)
        start_energies = np.array(list(start_energies_dict.values()))
        end_energies = np.array(list(end_energies_dict.values()))
        change_energies = np.array(list(change_energies_dict.values()))
        start_energy = start_energies.sum()
        end_energy = end_energies.sum()
        change_energy = change_energies.sum()

        start_misplaceds = start_energies > 0.0 
        end_misplaceds = end_energies > 0.0
        changeds = change_energies > 0.0

        num_broken = sum(curr_obj['isBroken'] for curr_obj in self.curr_objs_id_to_pose.values())
        num_initially_misplaced = start_misplaceds.sum()
        num_fixed = num_initially_misplaced - (start_misplaceds & end_misplaceds).sum()
        num_newly_misplaced = (end_misplaceds & np.logical_not(start_misplaceds)).sum()

        prop_fixed = 1.0 if num_initially_misplaced == 0 else num_fixed / num_initially_misplaced

        metrics = {
            f'{self.curr_stage}/start_energy': start_energy,
            f'{self.curr_stage}/end_energy': end_energy,
            f'{self.curr_stage}/success': float(end_energy == 0),
            f'{self.curr_stage}/prop_fixed': prop_fixed,
            f'{self.curr_stage}/prop_fixed_strict': float((num_newly_misplaced == 0) * prop_fixed),
            f'{self.curr_stage}/prop_misplaced': end_misplaceds.sum()/num_initially_misplaced if num_initially_misplaced > 0 else None,
            f'{self.curr_stage}/energy_prop': end_energy/start_energy if start_energy > 0 else None,
            f'{self.curr_stage}/num_misplaced': end_misplaceds.sum(),
            f'{self.curr_stage}/num_newly_misplaced': num_newly_misplaced.sum(),
            f'{self.curr_stage}/num_initially_misplaced': num_initially_misplaced,
            f'{self.curr_stage}/num_fixed': num_fixed.sum(),
            f'{self.curr_stage}/num_broken': num_broken,
            f'{self.curr_stage}/change_energy': change_energy,
            f'{self.curr_stage}/num_changed': changeds.sum(),
        }

        newly_misplaced = end_misplaceds & np.logical_not(start_misplaceds)
        where_misplaced = np.where(end_misplaceds)[0]
        missed_obj = [list(end_energies_dict.keys())[i].split('|')[0]+f'_nm={newly_misplaced[i]}' for i in list(where_misplaced)]

        return metrics, missed_obj, end_energies_dict




            
        

    
import cv2, ctypes, logging, os, numpy as np, pickle
from numpy import ma
from collections import OrderedDict
from skimage.morphology import binary_closing, disk
import scipy, skfmm
import matplotlib.pyplot as plt


from rearrange_on_proc.arguments import args
from rearrange_on_proc.utils.utils import pool2d
from rearrange_on_proc.utils.utils import dijkstra
from rearrange_on_proc.utils.utils import visdomImage
import pdb
import json
from rearrange_on_proc.utils.utils import MyEncoder,visdomImage
from numpy.lib.stride_tricks import as_strided
from rearrange_on_proc.constants import ROTATE_LEFT,ROTATE_RIGHT,MOVE_AHEAD,MOVE_LEFT,MOVE_RIGHT,MOVE_BACK

class FMMPlanner():
    def __init__(self, traversible, num_rots, step_size, obstructed_actions, visdom):
        self.traversible = traversible
        self.angle_value = [0, -2.0*np.pi/num_rots, +2.0*np.pi/num_rots, 0]
        self.step_size_on_map = step_size
        self.num_rots = num_rots
        self.action_list = self.search_actions()
        self.obstructed_actions = obstructed_actions

        self.visdom = visdom
    
    # ? lwj: action_list = [[3], [1,3], [2,3], [1,3,3,3], [2,3,3,3], [1,1,3], [2,2,3], [1,1,3,3,3], [2,2,3,3,3]]
    def search_actions(self):
        action_list = [[3]]
        append_list_pos = []
        append_list_neg = []
        for i in range(self.num_rots//2):
            append_list_pos.append(1)
            append_list_neg.append(2)
            action_list.append(append_list_pos[:]+[3])
            action_list.append(append_list_neg[:]+[3])
            action_list.append(append_list_pos[:]+[3]+[3]+[3])
            action_list.append(append_list_neg[:]+[3]+[3]+[3])
        return action_list
    
    def set_goal(self, goal):
        # 该函数的作用其实主要是为了算从goal开始边界演化的距离dd(即改动self.fmm_dist)，lwj验证过dd_mask其实和traversible_ma差不多（可能只相差几个像素）
        traversible_ma = ma.masked_values(self.traversible*1, 0) #带mask的traversible，masked的点 会影响skfmm.distance计算
        goal_x, goal_y = int(goal[0]),int(goal[1])
        goal_x = min(goal_x, traversible_ma.shape[1]-1)
        goal_y = min(goal_y, traversible_ma.shape[0]-1)
        goal_x = max(goal_x, 0)
        goal_y = max(goal_y, 0)
        traversible_ma[goal_y, goal_x] = 0 #traversible把0先mask, 然后只将goal设为0
        dd = skfmm.distance(traversible_ma, dx=1) #Fast Marching Method是一个解决边界演化的算法，指定边界的演化速度函数（方向为边界点的法向量），即给定一个边界（设为0），会算出从边界到x的演化距离和时间
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan))) #把 masked的点设为NaN,
        dd = ma.filled(dd, np.max(dd)+1)
        self.fmm_dist = dd  #dd中masked的点设为最大值
        return dd_mask
    
    def _virtual_steps(self, u_list, state_x_z_theta, check_collision=True):
        traversible = self.traversible
        goal_dist = self.fmm_dist
        angle_value = self.angle_value #[0, -pi/2, pi/2, 0]
        boundary_limits = np.array(traversible.shape)[::-1]
        x, y, t = state_x_z_theta
        out_states = []
        # ?lwjlwj: 感觉这里y,x反了
        cost_start = goal_dist[int(y), int(x)] #Actual distance in cm.？lwj
        collision_reward = 0
        for i in range(len(u_list)):
            action = u_list[i]
            x_new, y_new, t_new = x*1., y*1. , t*1.
            if action == 3:
                angl = t
                x_new = x+np.cos(angl)*self.step_size_on_map
                y_new = y+np.sin(angl)*self.step_size_on_map
                t_new = angl
            elif action == 15:
                angl = t
                x_new = x-np.cos(angl)*self.step_size_on_map
                y_new = y-np.sin(angl)*self.step_size_on_map
                t_new = angl
            elif action == 16:
                angl = t
                x_new = x-np.sin(angl)*self.step_size_on_map
                y_new = y+np.cos(angl)*self.step_size_on_map
                t_new = angl
            elif action == 17:
                angl = t
                x_new = x+np.sin(angl)*self.step_size_on_map
                y_new = y-np.cos(angl)*self.step_size_on_map
                t_new = angl
            elif action > 0:
                t_new = t + angle_value[action]
            
            # action in [3, 15, 16, 17], 针对原rotation=0或180则对应 前后左右；对rotation=90或270, 则后前右左；
            
            collision_reward = -1
            inside = np.all(np.array([int(x_new), int(y_new)]) < np.array(boundary_limits))
            inside = inside and np.all(np.array([int(x_new),int(y_new)]) >= np.array([0,0]))
            _new_state = [x,y,t]
            # print(inside, action)
            if inside:
                not_collided = True
                if (action in [3, 15, 16, 17]) and check_collision:
                    for s in np.linspace(0, 1, self.step_size_on_map+5):
                        _x = x * s + (1-s) * x_new
                        _y = y * s + (1-s) * y_new
                        bounds = args.map_size//args.map_resolution
                        if  _x < bounds and _y < bounds:
                            not_collided = not_collided and traversible[int(_y), int(_x)]
                        if  _x < bounds and (_y-3) < bounds:
                            not_collided = not_collided and traversible[int(_y-3), int(_x)]
                        if  _x < bounds and (_y+3) < bounds:
                            not_collided = not_collided and traversible[int(_y+3), int(_x)]
                        if  _y < bounds and (_x-3) < bounds:
                            not_collided = not_collided and traversible[int(_y), int(_x-3)]
                        if  _y < bounds and (_x+3) < bounds:
                            not_collided = not_collided and traversible[int(_y), int(_x+3)]
                        if  (_y-3) < bounds and (_x-3) < bounds:
                            not_collided = not_collided and traversible[int(_y-3), int(_x-3)]
                        if  (_y-3) < bounds and (_x+3) < bounds:
                            not_collided = not_collided and traversible[int(_y-3), int(_x+3)]
                        if  (_y+3) < bounds and (_x-3) < bounds:
                            not_collided = not_collided and traversible[int(_y+3), int(_x-3)]
                        if  (_y+3) < bounds and (_x+3) < bounds:
                            not_collided = not_collided and traversible[int(_y+3), int(_x+3)]
                        if not_collided is False:
                            break
                if not_collided:
                    collision_reward = 0
                    x, y, t = x_new, y_new, t_new
                    _new_state = [x,y,t]
            else:
                break
            out_states.append(_new_state) #virtual step可以成功（不碰撞）的state

        if u_list in self.obstructed_actions:
            collision_reward = -2
        cost_end = goal_dist[int(y), int(x)]
        reward_near_goal = 0.
        if cost_end < self.step_size_on_map:
            reward_near_goal = 1.
        costs = (cost_end - cost_start)
        reward = -costs + reward_near_goal + collision_reward
        return reward, (out_states)

    def find_best_action_set(self, state_x_z_theta, spacious=False, multi_act=0):
        goal_dist = self.fmm_dist
        traversible = self.traversible
        action_list = self.action_list
        best_list = [3]
        max_margin = 0
        obst_dist = []
        best_reward, state_list = self._virtual_steps(best_list, state_x_z_theta)
        best_reward = best_reward+0.1
        max_margin_state = state_list
        max_margin_act = [0]
        feasible_acts = []
        feasible_states = []
        sm_cut_reward, sm_state_list = self._virtual_steps([3], state_x_z_theta)
        sm_cut_reward_zero, sm_state_list = self._virtual_steps([0], state_x_z_theta)
        sm_cut_reward = max(sm_cut_reward, sm_cut_reward_zero)
        smarter_acts = []
        smarter_states = []
        st_lsts, rews = [], []
        # ? lwj: action_list = [[3], [1,3], [2,3], [1,3,3,3], [2,3,3,3], [1,1,3], [2,2,3], [1,1,3,3,3], [2,2,3,3,3]]
        # [3]: 前/后选一个（与当前rotation有关），[1,3]:左转+前/后，[2,3]: 右转+前/后
        for a_list in action_list:
            rew, st_lst = self._virtual_steps(a_list, state_x_z_theta)
            # Prefer shorter action sequences.
            rew = rew - len(st_lst)*0.05
            rews.append(rew)
            st_lsts.append(st_lst)
            
            if rew > best_reward:
                best_list = a_list
                best_reward = rew
                state_list = (st_lst)
            if False: #rew > 4: #self.env.dilation_cutoff:
                current_margin = self.get_obst_dist(st_lst[-1])
                if current_margin > max_margin:
                    max_margin=current_margin
                    max_margin_state=st_lst
                    max_margin_act=a_list
            if rew > 0:
                feasible_acts.append(a_list)
                feasible_states.append(st_lst)
            if rew >= max(sm_cut_reward,0):
                if a_list == [0] and rew < 1:
                    continue #没有动，且离goal也在一步距离外
                smarter_acts.append(a_list)
                smarter_states.append(st_lst)
      
        if not (len(best_list) == len(state_list)):
            print(len(best_list),len(state_list))
            raise Exception("Not equal")
        if not spacious or (len(max_margin_act)==1 and max_margin_act[0]==0):
            # print(0, best_list, best_reward, np.array(rews))
            return best_list, state_list  #不够宽敞 -》返回最大奖励的action_list
        else:
            # print(1, max_margin_act, max_margin_state)
            return max_margin_act, max_margin_state  #lwj: 够宽敞 -》（根据前面注释掉的逻辑）返回 virtual能够走到的最后一步 距离障碍物最远？的action_list

    # ? lwj: 好像没有被用到
    def compare_goal(self, a, goal_dist):
        goal_dist = self.fmm_dist
        x,y,t = a
        cost_end = goal_dist[int(y), int(x)]
        dist = cost_end*1.
        if dist < self.step_size_on_map*1:
            return True
        return False

    # lwj: 整体感觉都是先靠virtual step先试几套动作，然后去选一个最好的（为何不直接根据fmm_distance规划呢，既然fmm计算的时候已经把目前已知的障碍物（？膨胀—）并且mask了）
    def get_action(self, state_x_z_theta):
        best_action_list, state_list = self.find_best_action_set(state_x_z_theta, False, 0)
        # pdb.set_trace()
        return best_action_list[0], state_list[0], best_action_list

    def get_action_sequence_dij(self,map_positon_xy:np.ndarray,rotation_init,point_goal:np.ndarray,held_mode:bool):
        '''
        根据dijkstra算法计算最短路径
        map_positon_xy：机器人在地图上的位置
        rotation：机器人的转向
        '''
        #1.对地图进行下采样
        step_pix = int(args.STEP_SIZE / args.map_resolution)
        traversible = self.traversible.astype(int)
        pooled_map = pool2d(traversible,kernel_size=step_pix,stride=step_pix,pool_mode='avg')
        pooled_map = (pooled_map > 0.8).astype(int)
        rows, cols = pooled_map.shape
        
        # visdomImage(pooled_map, self.visdom, tag='01')
        # import pdb
        # pdb.set_trace()
        
        #2.用dijkstra算法计算最短路径
        start = (min(map_positon_xy[1]//step_pix,rows-1),min(map_positon_xy[0]//step_pix,cols-1))
        goal = (min(point_goal[1]//step_pix,rows-1),min(point_goal[0]//step_pix,cols-1))
        path = dijkstra(pooled_map,start=start,end=goal)
        
        #3.根据最短路径获得动作序列
        action_list = []
        cur_state = start
        rotation = rotation_init
        if path != None:
            for i in range(1,len(path)):
                next_state = path[i]
                if not held_mode:
                    acts,rotation = self.choose_acts(cur=cur_state,next=next_state,cur_rotation=rotation)
                else:
                    acts,rotation = self.choose_acts_when_held_obj(pooled_map=pooled_map,cur=cur_state,next=next_state,cur_rotation=rotation)
                action_list = action_list + acts
                cur_state = next_state
        #如果规划不出来就用启发式的方法
        else:
            theta = -np.deg2rad(rotation) + np.pi / 2
            a, state, action_list = self.get_action(np.array([map_positon_xy[0], map_positon_xy[1], theta]))
        # print("action_list",action_list)
        return action_list,path

    def check_empty_on_right(self,pooled_map,cur_pos,cur_rotation)->bool:
        '''检查机器人右边有没有障碍物'''
        cur_x = cur_pos[1]
        cur_y = cur_pos[0]
        max_x = pooled_map.shape[1] - 1
        max_y = pooled_map.shape[0] - 1
        min_x = 0
        min_y = 0
        if cur_rotation == 0:
            return pooled_map[cur_y][min(cur_x+1,max_x)] and pooled_map[cur_y][min(max_x,cur_x+2)]
        elif cur_rotation == 90:
            return pooled_map[max(min_y,cur_y-1)][cur_x] and pooled_map[max(min_y,cur_y-2)][cur_x]
        elif cur_rotation == 180:
            return pooled_map[cur_y][max(cur_x-1,min_x)] and pooled_map[cur_y][max(min_x,cur_x-2)]
        else:
            return pooled_map[min(max_y,cur_y+1)][cur_x] and pooled_map[min(max_y,cur_y+2)][cur_x]
        
    def check_empty_on_left(self,pooled_map,cur_pos,cur_rotation)->bool:
        '''检查机器人左边有没有障碍物'''
        cur_x = cur_pos[1]
        cur_y = cur_pos[0]
        max_x = pooled_map.shape[1] - 1
        max_y = pooled_map.shape[0] - 1
        min_x = 0
        min_y = 0
        if cur_rotation == 0:
            return pooled_map[cur_y][max(cur_x-1,min_x)] and pooled_map[cur_y][max(min_x,cur_x-2)]
        elif cur_rotation == 90:
            return pooled_map[min(max_y,cur_y+1)][cur_x] and pooled_map[min(max_y,cur_y+2)][cur_x]
        elif cur_rotation == 180:
            return pooled_map[cur_y][min(cur_x+1,max_x)] and pooled_map[cur_y][min(max_x,cur_x+2)]
        else:
            return pooled_map[max(min_y,cur_y-1)][cur_x] and pooled_map[max(min_y,cur_y-2)][cur_x]
    
    def check_empty_ahead(self,pooled_map,cur_pos,cur_rotation)->bool:
        '''检查机器人前边有没有障碍物'''
        cur_x = cur_pos[1]
        cur_y = cur_pos[0]
        max_x = pooled_map.shape[1] - 1
        max_y = pooled_map.shape[0] - 1 
        min_x = 0
        min_y = 0
        if cur_rotation == 0:
            return pooled_map[min(max_y,cur_y+1)][cur_x] and pooled_map[min(max_y,cur_y+2)][cur_x]
        elif cur_rotation == 90:
            return pooled_map[cur_y][min(cur_x+1,max_x)] and pooled_map[cur_y][min(max_x,cur_x+2)]
        elif cur_rotation == 180:
            return pooled_map[max(min_y,cur_y-1)][cur_x] and pooled_map[max(min_y,cur_y-2)][cur_x]
        else:
            return pooled_map[cur_y][max(cur_x-1,min_x)] and pooled_map[cur_y][max(min_x,cur_x-2)]
    
    def acquire_next_direction(self,dx,dy,rotation):
        '''判断路径的下一点在机器人的哪个方向'''
        if dx == 1 and dy ==0:
            if rotation == 0:
                direction = "right"
            elif rotation == 270:
                direction = "back"
            elif rotation == 180:
                direction = "left"
            else:
                direction = "ahead"
        elif dx == -1 and dy == 0:
            if rotation == 0:
                direction = "left"
            elif rotation == 90:
                direction = "back"
            elif rotation == 180:
                direction = "right"
            else:
                direction = "ahead"
        elif dx == 0 and dy == 1:
            if rotation == 90:
                direction = "left"
            elif rotation == 180:
                direction = "back"
            elif rotation == 270:
                direction = "right"
            else:
                direction = "ahead"
        elif dx == 0 and dy == -1:
            if rotation == 0:
                direction = "back"
            elif rotation == 90:
                direction = "right"
            elif rotation == 270:
                direction = "left"
            else:
                direction = "ahead"
        return direction
    
    def choose_acts_when_held_obj(self,pooled_map,cur,next,cur_rotation):
        """
        根据当前坐标点和下一个坐标点生成一组动作,动作空间添加了MOVE_RIGHT,MOVE_LEFT
        """
        dx = next[1]-cur[1]
        dy = next[0]-cur[0]
        acts = []
        rotation = cur_rotation
        #判断下一步的方向
        direction = self.acquire_next_direction(dx,dy,rotation)
        #如果下一步在前面
        if direction == "ahead":
            #如果前面没有障碍物，前进
            if self.check_empty_ahead(pooled_map=pooled_map,cur_pos=cur,cur_rotation=rotation):
                acts.append(MOVE_AHEAD)
            #如果右边没有障碍物，右转左移
            elif self.check_empty_on_right(pooled_map=pooled_map,cur_pos=cur,cur_rotation=rotation):
                acts.append(ROTATE_RIGHT)
                acts.append(MOVE_LEFT)
                rotation += args.DT
            #如果左边没有障碍物，左转右移
            elif self.check_empty_on_left(pooled_map=pooled_map,cur_pos=cur,cur_rotation=rotation):
                acts.append(ROTATE_LEFT)
                acts.append(MOVE_RIGHT)
                rotation -= args.DT
            #如果前面左边右边都有障碍物，摆烂前进
            else:
                acts.append(MOVE_AHEAD)
        #如果下一步在右边
        elif direction == 'right':
            #如果右边没有障碍物，右转前进
            if self.check_empty_on_right(pooled_map=pooled_map,cur_pos=cur,cur_rotation=rotation):
                acts.append(ROTATE_RIGHT)
                acts.append(MOVE_AHEAD)
                rotation += args.DT
            #如果右边有障碍物，但是地图又规划了，说明一步之内没有障碍物，两步之内有，右移一下
            else:
                acts.append(MOVE_RIGHT)
        #如果下一步在左边
        elif direction == 'left':
            #如果左边没有障碍物，左转前进
            if self.check_empty_on_left(pooled_map=pooled_map,cur_pos=cur,cur_rotation=rotation):
                acts.append(ROTATE_LEFT)
                acts.append(MOVE_AHEAD)
                rotation -= args.DT
            #如果左边有障碍物，但是地图又规划了，说明一步之内没有障碍物，两步之内有，左移一下
            else:
                acts.append(MOVE_LEFT)
        #如果下一步在后边
        elif direction == "back":
            #如果左边没有障碍物先左转
            if self.check_empty_on_left(pooled_map=pooled_map,cur_pos=cur,cur_rotation=rotation):
                acts.append(ROTATE_LEFT)
                rotation -= args.DT
                #现在变成下一步在左边了
                if self.check_empty_on_left(pooled_map=pooled_map,cur_pos=cur,cur_rotation=rotation):
                    acts.append(ROTATE_LEFT)
                    acts.append(MOVE_AHEAD)
                    rotation -= args.DT
                else:
                    acts.append(MOVE_LEFT)
            #如果右边没有障碍物先右转
            elif self.check_empty_on_right(pooled_map=pooled_map,cur_pos=cur,cur_rotation=rotation):
                acts.append(ROTATE_RIGHT)
                rotation += args.DT
                #现在变成下一步在右边了
                if self.check_empty_on_right(pooled_map=pooled_map,cur_pos=cur,cur_rotation=rotation):
                    acts.append(ROTATE_RIGHT)
                    acts.append(MOVE_AHEAD)
                    rotation += args.DT
                else:
                    acts.append(MOVE_RIGHT)
            else:
                acts.append(MOVE_BACK)
        rotation = rotation % 360
        return acts,rotation


    def choose_acts(self,cur,next,cur_rotation):
        """
        根据当前坐标点和下一个坐标点生成一组动作
        """
        dx = next[1]-cur[1]
        dy = next[0]-cur[0]
        acts = []
        rotation = cur_rotation
        if dx == 1 and dy ==0:
            if rotation == 0:
                acts.append(ROTATE_RIGHT)
                rotation += args.DT
            elif rotation == 270:
                acts.append(ROTATE_RIGHT)
                acts.append(ROTATE_RIGHT)
                rotation += 2*args.DT
            elif rotation == 180:
                acts.append(ROTATE_LEFT)
                rotation -= args.DT
            acts.append(MOVE_AHEAD)
        elif dx == -1 and dy == 0:
            if rotation == 0:
                acts.append(ROTATE_LEFT)
                rotation -= args.DT
            elif rotation == 90:
                acts.append(ROTATE_LEFT)
                acts.append(ROTATE_LEFT)
                rotation -= 2*args.DT
            elif rotation == 180:
                acts.append(ROTATE_RIGHT)
                rotation += args.DT
            acts.append(MOVE_AHEAD)
        elif dx == 0 and dy == 1:
            if rotation == 90:
                acts.append(ROTATE_LEFT)
                rotation -= args.DT
            elif rotation == 180:
                acts.append(ROTATE_LEFT)
                acts.append(ROTATE_LEFT)
                rotation -= 2*args.DT
            elif rotation == 270:
                acts.append(ROTATE_RIGHT)
                rotation += args.DT
            acts.append(MOVE_AHEAD)
        elif dx == 0 and dy == -1:
            if rotation == 0:
                acts.append(ROTATE_RIGHT)
                acts.append(ROTATE_RIGHT)
                rotation += 2*args.DT
            elif rotation == 90:
                acts.append(ROTATE_RIGHT)
                rotation += args.DT
            elif rotation == 270:
                acts.append(ROTATE_LEFT)
                rotation -= args.DT
            acts.append(MOVE_AHEAD)
        rotation = rotation % 360
        return acts,rotation

    

def main():
    im = cv2.imread('after.png', cv2.IMREAD_UNCHANGED)
    # im = im > 127
    y,x = np.where(im)
    planner = FMMPlanner(im)
    fig, ax = plt.subplots()
    # ax = ax[::-1]
    rng = np.random.RandomState(1)
    goal_ind = rng.choice(y.size)
    start_ind = rng.choice(y.size)
    goal = [350, 222]
    state_x_z_theta = [317, 173, 59.881]
    planner.set_goal(goal)
    ax.imshow(im * 1., vmin=-0.5, vmax=1.5)
    ax.plot(goal[0], goal[1], 'rx')
    ax.plot(state_x_z_theta[0], state_x_z_theta[1], 'rs')
    states = []
    for i in range(1000):
        a, state_x_z_theta = planner.get_action(state_x_z_theta)
        states.append(state_x_z_theta)
        if a == 0:
            break
    states = np.array(states)
    ax.plot(states[:, 0], states[:, 1], 'r.')
    
    fig.savefig('fmm.png', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    main()
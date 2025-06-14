import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
# from rearrange_on_proc.arguments import args
import os
import torch
import pdb
import skimage.morphology
from ivn_aff.mapper import Mapper
# from rearrange_on_proc.planner import FMMPlanner SEGMENTATION_CATEGORY_to_COLOR CATEGORY_LIST
from ivn_aff.constants import CATEGORY_PALETTE, OTHER_PALETTE
from PIL import Image
# from rearrange_on_proc.utils.utils import pool2d, RGB_to_Hex, get_colormap
# from rearrange_on_proc.utils.aithor import  get_origin_T_camX, get_3dbox_in_geom_format, get_amodal2d
import math
import time

fov = 90.0
map_resolution = 0.05
STEP_SIZE = 0.25

class Animation():
    '''
    util for generating movies of the agent and TIDEE modules
    '''

    def __init__(self, W,H, name_to_id=None):  

        # self.fig = plt.figure(1, figsize=(30, 15))
        plt.figure(figsize=(10,10))
        plt.clf()

        self.W = W
        self.H = H

        self.name_to_id = name_to_id

        self.object_tracker = None
        self.camX0_T_origin = None

        self.image_plots = []

        self.demo_path = '/home/yyl/lwj/rearrange_on_ProcTHOR/demo_train'
        self.rgb_dir = '5framework_rgb'
        self.depth_dir = '5framework_depth'
        self.rgbseg_dir = '5framework_rgbseg'
        self.dMap_dir = '5framework_dMap'
        self.segMap_dir = '5framework_segMap'
        for dir in [self.rgb_dir, self.depth_dir, self.rgbseg_dir, self.dMap_dir, self.segMap_dir]:
            if not os.path.exists(os.path.join(self.demo_path, dir)):
                os.mkdir(os.path.join(self.demo_path, dir))


    def add_frame(self, image:np.ndarray, depth, seg_image, mapper:Mapper, add_map:bool, add_map_seg: bool, selem, selem_agent_radius, point_goal:list = None,path=None,stage = None, step=None,text=None,action=None):
        
        
        plt.clf()
        # self.fig.clear()
        ax = []
        step_pix = int(STEP_SIZE / map_resolution)

        simple_version = False
        if simple_version:
            ncols = 3
            spec = gridspec.GridSpec(ncols=3, nrows=1,
                    figure=self.fig, left=0., right=1., wspace=0.0001, hspace=0.0001)
        else:
            ncols = 4
        for a in ax:
            a.axis('off')
        if step is not None:
            plt.title(f"{stage} stage -- step:{step}: {action}")
        else:
            plt.title(f"{action}")
    
        if simple_version:
            # plt.imshow(image)
            if seg_image is not None:
                seg_image = seg_image.astype(np.uint8)
                # plt.imshow(seg_image)
                ax[0].imshow(np.flip(seg_image, axis=0), origin='lower')
                ax[0].set_title('rgb_seg')
                # plt.set_title('segmentation')
            else:
                image = image.astype(np.uint8)
                ax[0].imshow(np.flip(image, axis=0), origin='lower')
                ax[0].set_title('rgb')
            if add_map_seg:
                map_topdown_seg = mapper.get_topdown_semantic_map()
                explored = mapper.get_explored_map(selem, point_count=1) #机器人去过的地方 or map有点的地方
                obstacle = mapper.get_obstacle()    #map上点的数量大于100的记为障碍物
                visited = mapper.get_visited()
                map_topdown_seg_vis = visualize_topdownSemanticMap(map_topdown_seg, explored, obstacle, visited)

                state_xy = mapper.get_position_on_map()
                state_theta = mapper.get_rotation_on_map()
                arrow_len = 1.0/mapper.resolution
                ax[1].imshow(map_topdown_seg_vis, origin = 'lower')
                ax[1].arrow(state_xy[0], state_xy[1], 
                        arrow_len*np.cos(state_theta+np.pi/2),
                        arrow_len*np.sin(state_theta+np.pi/2), 
                        color='b', head_width=10)
                if point_goal is not None:
                    ax[1].plot(point_goal[0], point_goal[1], 'bo',markersize = 8)
                
                if path != None:
                    for i in range(len(path)):
                        x,y = path[i]
                        x = x*step_pix
                        y = y*step_pix
                        ax[1].plot(y, x, 'bo',markersize = 4)
                # if text is not None:
                #     ax[1].set_title(text)
                ax[1].set_title('seg_map (top down)')
            if add_map:
                m_vis = np.invert(mapper.get_traversible_map(
                    selem_agent_radius, 1,loc_on_map_traversible=False))
                ax[2].imshow(m_vis, origin='lower', vmin=0, vmax=1,
                        cmap='Greys')
                state_xy = mapper.get_position_on_map()
                state_theta = mapper.get_rotation_on_map()
                arrow_len = 1.0/mapper.resolution
                ax[2].arrow(state_xy[0], state_xy[1], 
                            arrow_len*np.cos(state_theta+np.pi/2),
                            arrow_len*np.sin(state_theta+np.pi/2), 
                            color='b', head_width=10)
                
                if point_goal is not None:
                    ax[2].plot(point_goal[0], point_goal[1], color='blue', marker='o',linewidth=10, markersize=6)

                # ax[3].imshow(m_vis, origin='lower', vmin=0, vmax=1,
                #          cmap='Greys')
                if path != None:
                    for i in range(len(path)):
                        x,y = path[i]
                        x = x*step_pix
                        y = y*step_pix
                        ax[2].plot(y, x, 'ro',markersize = 4)
            
                ax[2].plot(state_xy[0], state_xy[1], 'go',markersize = 8)
                if point_goal is not None:
                    ax[2].plot(point_goal[0], point_goal[1], 'bo',markersize = 8)
                if text is not None:
                    ax[2].set_title(text)
                else:
                    ax[2].set_title("path")
        else:
            image = image.astype(np.uint8)
            plt.imshow(image)
            
            plt.axis('off')
            # plt.set_frame_on(False)
            plt.xticks([])
            # file_path = os.path.join(self.demo_path, self.rgb_dir, f"{stage}_{step}_{action}.png" )
            # plt.savefig(file_path, bbox_inches='tight', pad_inches =-0.1)

            plt.clf()
            if add_map:
                m_vis = np.invert(mapper.get_traversible_map(
                    selem_agent_radius, 1,loc_on_map_traversible=True))
                plt.imshow(m_vis * 0.8, origin='lower', vmin=0, vmax=1, cmap='Greys')
                map_for_distance = mapper.get_map_for_view_distance()
                norm = plt.Normalize(vmin = 0, vmax = 5)
                plt.imshow(map_for_distance, alpha = 0.7, origin='lower', interpolation='nearest', cmap=plt.get_cmap('jet'), norm = norm)
                state_xy = mapper.get_position_on_map()
                state_theta = mapper.get_rotation_on_map()
                arrow_len = 1.0/mapper.resolution
                plt.arrow(state_xy[0], state_xy[1], 
                            arrow_len*np.cos(state_theta+np.pi/2),
                            arrow_len*np.sin(state_theta+np.pi/2), 
                            color='b', head_width=10)
                
                if point_goal is not None:
                    plt.plot(point_goal[1], point_goal[0], color='blue', marker='o',linewidth=10, markersize=6)

                # ax[3].imshow(m_vis, origin='lower', vmin=0, vmax=1,
                #          cmap='Greys')
                if path != None:
                    for i in range(len(path)):
                        x,y = path[i]
                        x = x*step_pix
                        y = y*step_pix
                        plt.plot(y, x, 'ro',markersize = 4)
                # print('&&&&&&&&&&', state_xy, path)

                plt.plot(state_xy[0], state_xy[1], 'go',markersize = 8)
                if point_goal is not None:
                    plt.plot(point_goal[1], point_goal[0], 'bo',markersize = 8)
                # if text is not None:
                #     ax[1].set_title(text)
                # else:
                #     ax[1].set_title("path")
                plt.axis('off')
                # plt.set_frame_on(False)
                plt.xticks([])

            if seg_image is not None:
                plt.clf()
                seg_image = seg_image.astype(np.uint8)
                plt.imshow(seg_image)
                # ax[4].set_title('segmentation')
                plt.axis('off')
                # plt.set_frame_on(False)
                plt.xticks([])
                # file_path = os.path.join(self.demo_path, self.rgbseg_dir, f"{stage}_{step}_{action}.png" )
                # plt.savefig(file_path, bbox_inches='tight', pad_inches =-0.1)

                if add_map_seg:
                    plt.clf()
                    map_topdown_seg = mapper.get_topdown_semantic_map()
                    explored = mapper.get_explored_map(selem, point_count=1) #机器人去过的地方 or map有点的地方
                    obstacle = mapper.get_obstacle()    #map上点的数量大于100的记为障碍物
                    visited = mapper.get_visited()
                    map_topdown_seg_vis = visualize_topdownSemanticMap(map_topdown_seg, explored, obstacle, visited)
                    plt.imshow(map_topdown_seg_vis, origin = 'lower')
                    plt.arrow(state_xy[0], state_xy[1], 
                            arrow_len*np.cos(state_theta+np.pi/2),
                            arrow_len*np.sin(state_theta+np.pi/2), 
                            color='b', head_width=10)
                    if point_goal is not None:
                        plt.plot(point_goal[1], point_goal[0], 'bo',markersize = 8)
                    
                    if path != None:
                        for i in range(len(path)):
                            x,y = path[i]
                            x = x*step_pix
                            y = y*step_pix
                            plt.plot(y, x, 'bo',markersize = 2)
                    # ax[3].set_title('seg_map (top down)')
                    plt.axis('off')
                    # plt.set_frame_on(False)
                    plt.xticks([])
            
            if depth is not None:
                d = depth * 1.
                d[d > 10] = 0
                # d = normalization(d) * 255
                d = d / 10 * 255
                d = d.astype(np.uint8)
                # d_color = cv2.applyColorMap(d, 2)
                # d_color = Image.fromarray(d_color)
                plt.clf()
                plt.imshow(d, cmap='gray')

                plt.axis('off')
                # plt.set_frame_on(False)
                plt.xticks([])
                # file_path = os.path.join(self.demo_path, self.depth_dir, f"{stage}_{step}_{action}.png" )
                # plt.savefig(file_path, bbox_inches='tight', pad_inches =-0.1)

        canvas = FigureCanvas(plt.gcf())

        canvas.draw()       # draw the canvas, cache the renderer
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        # visdom.matplot(plt)        
        # pdb.set_trace()
        self.image_plots.append(image)

    def render_movie(self, dir, episode, process_id='', tag='', fps=5):

        if not os.path.exists(dir):
            os.mkdir(dir)
        video_name = os.path.join(dir, f'output-pid{process_id}-task{episode}|{tag}.mp4')
        print(f"rendering to {video_name}")
        height, width, _ = self.image_plots[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_name, fourcc, 4, (width, height))

        # print("------------------again__------------", len(self.image_plots))
        for im in self.image_plots:
            rgb = np.array(im).astype(np.uint8)
            bgr = rgb[:,:,[2,1,0]]
            video_writer.write(bgr)

        cv2.destroyAllWindows()
        video_writer.release()

def visualize_segmentationRGB(rgb, segmented_dict = None, visualize_sem_seg = True, bbox = None):
    from detectron2.utils.visualizer import ColorMode, Visualizer
    from detectron2.structures import Boxes, Instances

    visualizer = Visualizer(rgb, instance_mode=ColorMode.IMAGE)
    H, W = rgb.shape[:2]
    if visualize_sem_seg:
        score_list = []
        category_list = [] 
        mask_list = []
        # for seg_category, seg_instance in segmented_dict.items():
            # for seg_instance_id, seg_instance in seg_instances.items():
        for i in range(len(segmented_dict['scores'])):
            score_list.append(segmented_dict['scores'][i])
            category_list.append(segmented_dict['categories'][i])
            mask_list.append(segmented_dict['masks'][i])
        v_outputs = Instances(
        image_size=(H, W),
        scores=np.array(score_list),
        pred_classes=np.array(category_list),
        pred_masks=np.array(mask_list),
        )
        vis_output = visualizer.draw_instance_predictions(v_outputs.to("cpu"))
        
    else:
        instances = Instances((H, W)).to(torch.device("cpu"))
        boxes = Boxes(bbox.to(torch.device("cpu")))
        instances.set('pred_boxes', boxes)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
    
    return vis_output.get_image()

def visualize_topdownSemanticMap(topdown_seg_map, explored, obstacle, visited):
    shape = topdown_seg_map.shape  #[H, W]
    color_palette = OTHER_PALETTE + CATEGORY_PALETTE
    topdown_seg_map = skimage.morphology.dilation(topdown_seg_map, skimage.morphology.disk(1))
    no_category_mask = topdown_seg_map == 0
    topdown_seg_map += 3  #之前类别1为分割第一个类，现在对应调色板的index 4
    # add the fourth channel (0: out-of-bounds or no category, 1: Floor or explored , 2: Wall (Obstacle without category), 3: visited)
    topdown_seg_map[no_category_mask] = 0
    mask = np.logical_and(no_category_mask, explored)
    topdown_seg_map[mask] = 1
    obstacle = skimage.morphology.binary_dilation(obstacle,  skimage.morphology.disk(1)) == True    
    mask = np.logical_and(no_category_mask, obstacle)
    topdown_seg_map[mask] = 2
    # visited = skimage.morphology.dilation(visited, skimage.morphology.disk(2))
    topdown_seg_map[visited] = 3

    # rgb 从0-1转换到0-255
    color_palette = [int(x * 255.) for x in color_palette]
    color_palette = np.uint8(color_palette).tolist()
    semantic_map = Image.new("P", (shape[1],shape[0]))
    semantic_map.putpalette(color_palette)
    semantic_map.putdata((topdown_seg_map.flatten()).astype(np.uint8))
    semantic_map = semantic_map.convert("RGBA")

    return semantic_map

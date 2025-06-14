import argparse
import numpy as np
# from rearrange_on_proc.constants import CATEGORY_LIST_LENGTH
parser = argparse.ArgumentParser()

parser.add_argument("--STEP_SIZE", type=int, default=0.25,
                    help="yaw movement delta")
parser.add_argument("--fov", type=int, default=90, help="field of view")
parser.add_argument("--W", type=int, default=480, help="image width")
parser.add_argument("--H", type=int, default=480, help="image height")
# parser.add_argument("--num_categories", type=int, default = CATEGORY_LIST_LENGTH, help='number of segmentation categories')
# 抬头角度
parser.add_argument("--HORIZON_DT", type=int, default=30,
                    help="pitch movement delta")
# 旋转角度
parser.add_argument("--DT", type=int, default=90, help="yaw movement delta")


parser.add_argument("--map_size",type=int,default=40,help="map size of room (m)")
parser.add_argument("--map_resolution",type=float,default=0.05,help="map resolution of room, XXX m/pixel")
parser.add_argument("--step_max",type=int,default=200,help="max step in exploring")
parser.add_argument("--rearrange_step_max", type=int, default=100, help="max step in rearrange navigation")
parser.add_argument("--use_seg",action="store_true", default=False, help='take segmentation and map and visualize')
parser.add_argument("--use_GT_seg", default=False, help='use GT segmentation')
parser.add_argument("--seg_utils",default="GT",type=str,help='which seg utils to be used,available options:["GT","MaskRCNN","yolov7"]')

#是否可视化
parser.add_argument("--create_movie", action="store_true", default=False, help="create mp4 movie")
parser.add_argument("--generate_rearrangement_images", action="store_true", default=False, help="generate obj images")

#可视化分辨率
parser.add_argument("--dpi", type=int, default=150, help="dpi for matplotlib. higher value will give higher res movies but slower to create.")
parser.add_argument("--movie_dir", type=str, default="./movies", help="where to output rendered movies")

parser.add_argument("--split", type=str, default="debug", help="which part to evaluate")
parser.add_argument("--areaRanges", type=str, default="all", help='which area range to evaluate, use "|" to split each option, available options: ["all", "<10", "10-60", "60-150", "150-300", ">300"]')
parser.add_argument("--roomNum", type=str, default="all", help='which roomNum to evaluate, use "|" to split each option, available options: ["all", "1", "2", "3", "4"]')


parser.add_argument("--load_submission", action="store_true", default=False, help="pick up where left off")

#探索阶段使用离线环境
parser.add_argument("--use_offline_ai2thor", action="store_true", default=False)

# 用于生成数据集的程序datagen_runner.py
parser.add_argument("--debug", "-d", action="store_true", default=False)
parser.add_argument("--train_unseen", "-t", action="store_true", default=False)
parser.add_argument("--choose_split", type=str, default='test', help='which split dataset to generate')
parser.add_argument("--x_display",type=int, default=2, help="x server port")
parser.add_argument("--only_generate_8_and_9", action="store_true", default=False)
#用关系变化判断移动
parser.add_argument("--dissimilar_threshold", type=float, default=0.35, help="threshold for percent of relations for object to be out of place")
parser.add_argument("--thresh_num_dissimilar", type=int, default=-1, help="threshold for minimum number of dissimilar for object to be out of place")

#用于保存explorer类
parser.add_argument("--store_walkthrough_explorer",action="store_true",default=False,help="need store walkthrough explorer")
parser.add_argument("--load_walkthrough_explorer",action="store_true",default=False,help="need load walkthrough explorer")
parser.add_argument("--store_unshuffle_explorer",action="store_true",default =False,help="need store unshuffle explorer")
parser.add_argument("--load_unshuffle_explorer",action="store_true",default=False,help="need load unshuffle explorer")
parser.add_argument("--store_explorer_path",type=str,default="../rearrange_on_ProcTHOR/rearrange_on_proc/explorer_data",help="")

#用于测试
parser.add_argument("--test_mode",type=str,default="only_walkthrough",help="which mode to test,available options:['two_phase','only_walkthrough','only_walkthrough_steps']")
parser.add_argument("--only_walkthrough_mode_max_steps",type=int,default=300)
parser.add_argument("--walkthrough_search",type=str,default="cover",help="walkthrough search policy,available options:['cover','cover_nearest', 'cover_continue','minViewDistance','mass']")
parser.add_argument("--unshuffle_search",type=str,default="cover",help="unshuffle search policy,available options:['cover','cover_continue','cover_nearest', 'same', 'localAuto','minViewDistance','mass']")
parser.add_argument("--unshuffle_match",type=str,default="feature_IoU_based",help="unshuffle match policy,available options:['feature_IoU_based', 'feature_centroid_based', 'feature_relation_based']")
parser.add_argument("--unshuffle_reorder",type=str,default="or_tools",help="unshuffle match policy,available options:['greedy', 'or_tools', 'random']")
parser.add_argument("--solution_str",type=str,default="")

#用于测试mass
parser.add_argument("--mass_modal_path",type=str,default="/home/yyl/lwj/rearrange_on_ProcTHOR/rearrange_on_proc/policy.pth")
parser.add_argument("--mass_device",type=str,default="1")

#用于yolov7
parser.add_argument("--device",type=str,default="0",help="which gpu to be used for yolov7 segmentation")

#用于测试加速
parser.add_argument("--debug_print", action="store_true", default=False)

obstacle_types = ['ArmChair', 'DogBed', 'Box', 'Chair', 'Desk', 'DiningTable', 'SideTable', 'Sofa', 'Stool', 'Television', 'Pillow', 'Bread', 'Apple', 'AlarmClock', 'Lettuce', 'GarbageCan', 'Laptop', 'Microwave', 'Pot', 'Tomato']
parser.add_argument("--obstacle_types", default=obstacle_types)

args = parser.parse_args()

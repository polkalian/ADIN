import numpy as np
import pdb
from constants import CATEGORY_to_ID, CATEGORY_LIST, INSTANCE_SEG_THRESHOLD
# from arguments import args
import sys
import os
import cv2
import torch
import torchvision.models 
import torchvision.transforms as transforms
import torchvision.ops as ops
import torch.nn as nn
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# from Segmentation.segmentation_helper import MASK_RCNN_SegmentationHelper
from yolov7.yolov7_seg import YOLOV7_SegmentationHelper

use_GT_seg = True
seg_utils = 'GT'
H = 224
W = 224
num_categories = 20


class SegmentationHelper():
    def __init__(self, controller) -> None:
        self.controller = controller
        self.use_GT_seg = use_GT_seg
        self.seg_utils = seg_utils

        # self.objId_to_color = controller.last_event.object_id_to_color
        # self.color_to_objId = controller.last_event.color_to_object_id
        if self.seg_utils == 'GT':           
            self.segmentation_helper = None
        # elif self.seg_utils == 'MaskRCNN':
        #     self.segmentation_helper = MASK_RCNN_SegmentationHelper()
        elif self.seg_utils == 'yolov7':
            self.segmentation_helper = YOLOV7_SegmentationHelper()
        
        # resnet18用于提取特征
        import ssl
 

        ssl._create_default_https_context = ssl._create_unverified_context

        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        self.feature_extractor=nn.Sequential(*list(self.feature_extractor.children())[:-4])
        self.feature_extractor.eval()
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    def get_seg_pred(self, rgb, isPickOrPutStage = False):
        '''
        Input: RGB
        Output: Segmentation: H * W * (num_category + 1), Segmented_dict:{'scores':, 'categories', 'masks'} 
        '''
        instance_seg = np.zeros((H, W, num_categories+1))
        segmented_dict = {}
        # segmented_dict = {
        #  '{categoryA}': {
        #     '{category}_{id1}': {
        #         'score': xxx,
        #         'mask': xxxx,
        #         'feature': xxxxx,      
        #      },
        #     '{category}_{id2}': {
        #         'score': xxx,
        #         'mask': xxxx,
        #         'feature': xxxxx,     
        #      }
        #  },
        #  '{categoryB}': {
        #     {...}, {...}
        #   }
        # }
        count_curr_instances_of_same_category = {}

        #提取特征
        # 采用Resent18
        rgb = Image.fromarray(rgb)
        height, width = rgb.size
        img_tensor = self.transform(rgb).unsqueeze(0)
        features = self.feature_extractor(img_tensor)

        # pdb.set_trace()
        if self.seg_utils == 'GT':           
            instance_masks = self.controller.last_event.instance_masks

            #采用fine-tune的YOLOv7【特征效果不好】
            # result, feature_map_list, im0s, im = self.segmentation_helper.seg_pred(rgb)

            for objectId, mask in instance_masks.items():
                category = objectId.split('|')[0]
                if category in CATEGORY_LIST:
                    if category not in count_curr_instances_of_same_category:
                        count_curr_instances_of_same_category[category] = 1
                    else:
                        count_curr_instances_of_same_category[category] += 1
                    category_id = CATEGORY_to_ID[category]

                    y_list, x_list = np.where(mask)
                    if len(x_list) == 0 or len(y_list) == 0:
                        # print(category)
                        continue
                    # instance_seg[:,:,category_id] += mask.astype('float')  #补充了+，考虑到同一视角下有同类别的多实例物体
                    instance_seg[:,:,category_id][mask.astype(bool)] = count_curr_instances_of_same_category[category]  #同一类别下的多实例编号从1开始
                    if category not in segmented_dict:
                        segmented_dict[category] = {}

                    bbox = torch.tensor([[min(x_list), min(y_list), max(x_list), max(y_list)]])
                    x_min,y_min,x_max,y_max = bbox[0]
                    if x_max == x_min:
                        x_max += 1
                    if y_max == y_min:
                        y_max += 1
                    adjust_bbox = (x_min, y_min, x_max, y_max)
                    normalized_bbox = torch.zeros_like(bbox, dtype=features.dtype)
                    normalized_bbox[0, 0] = bbox[0, 0] / width
                    normalized_bbox[0, 1] = bbox[0, 1] / height
                    normalized_bbox[0, 2] = bbox[0, 2] / width
                    normalized_bbox[0, 3] = bbox[0, 3] / height
            
                    ins_visual_feature = ops.roi_pool(features, [normalized_bbox], output_size=(1,1)).ravel()
                    ins_color_feature = self.get_color_feature(rgb, adjust_bbox)

                    concat_feature = torch.cat((ins_visual_feature, ins_color_feature)).ravel().cpu().detach().numpy()
                    
                    instance_curr_id = str(category) + '_' + str(count_curr_instances_of_same_category[category])
                    instance_info = {
                        'score': 1.0,
                        'mask': mask.astype(bool),
                        'feature': concat_feature,
                        'simulator_id':objectId
                    }
                    segmented_dict[category][instance_curr_id] = instance_info

        # elif self.seg_utils == 'MaskRCNN':
        #     # obj_feature还没获取
        #     origin_segmented_dict = self.segmentation_helper.get_instance_mask_seg_alfworld_both(self.controller)
        #     small_segmented_dict = origin_segmented_dict['small']
        #     large_segmented_dict = origin_segmented_dict['large']
        #     for i in range(len(small_segmented_dict['classes'])):
        #         category = small_segmented_dict['classes'][i]
        #         if category in CATEGORY_LIST:
        #             if category not in count_curr_instances_of_same_category:
        #                 count_curr_instances_of_same_category[category] = 1
        #             else:
        #                 count_curr_instances_of_same_category[category] += 1
        #             category_id = CATEGORY_to_ID[category]
        #             mask = small_segmented_dict['masks'][i]
        #             # instance_seg[:,:,category_id] = mask.astype('float')
        #             instance_seg[:,:,category_id][mask.astype(bool)] = count_curr_instances_of_same_category[category]  #同一类别下的多实例编号从1开始
        #             if category not in segmented_dict:
        #                 segmented_dict[category] = {}
        #
        #             instance_curr_id = str(category) + '_' + str(count_curr_instances_of_same_category[category])
        #             instance_info = {
        #                 'score': 1.0,
        #                 'mask': mask.astype(bool),
        #                 'feature': np.array(obj_feature),
        #                 'simulator_id': instance_curr_id
        #             }
        #             segmented_dict[category][instance_curr_id] = instance_info
        #
        #     for i in range(len(large_segmented_dict['classes'])):
        #         category = large_segmented_dict['classes'][i]
        #         if category in CATEGORY_LIST:
        #             if category not in count_curr_instances_of_same_category:
        #                 count_curr_instances_of_same_category[category] = 1
        #             else:
        #                 count_curr_instances_of_same_category[category] += 1
        #             category_id = CATEGORY_to_ID[category]
        #             mask = large_segmented_dict['masks'][i]
        #             # instance_seg[:,:,category_id] = mask.astype('float')
        #             instance_seg[:,:,category_id][mask.astype(bool)] = count_curr_instances_of_same_category[category]  #同一类别下的多实例编号从1开始
        #             if category not in segmented_dict:
        #                 segmented_dict[category] = {}
        #
        #             instance_curr_id = str(category) + '_' + str(count_curr_instances_of_same_category[category])
        #             instance_info = {
        #                 'score': 1.0,
        #                 'mask': mask.astype(bool),
        #                 'feature': np.array(obj_feature),
        #                 'simulator_id': instance_curr_id
        #             }
        #             segmented_dict[category][instance_curr_id] = instance_info
        
        elif self.seg_utils == 'yolov7':
            segmented_list = self.segmentation_helper.seg_pred(self.controller.last_event.frame)

            for obj in segmented_list:
                if obj['score'] < INSTANCE_SEG_THRESHOLD:
                    continue
                category = obj["label"]
                if category in CATEGORY_LIST:
                    if category not in count_curr_instances_of_same_category:
                        count_curr_instances_of_same_category[category] = 1
                    else:
                        count_curr_instances_of_same_category[category] += 1
                    category_id = CATEGORY_to_ID[category]
                    mask = obj['mask']
                    # mask = mask.astype(np.uint8)
                    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
                    # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, borderType = cv2.BORDER_CONSTANT,borderValue= 0)
                    y_list, x_list = np.where(mask)
                    if len(x_list) == 0 or len(y_list) == 0:
                        # print(category)
                        continue
                    bbox = torch.tensor([[min(x_list), min(y_list), max(x_list), max(y_list)]])
                    
                    if not isPickOrPutStage:
                        if category in CATEGORY_LIST[:60]:
                            if args.walkthrough_search == "minViewDistance" or args.unshuffle_search == "minViewDistance":
                                if bbox[0,0]/width < 0.05 or bbox[0,2]/width >0.95 or bbox[0,1]/height < 0.05 or bbox[0,3]/height > 0.95:
                                    continue
                    # instance_seg[:,:,category_id] += mask.astype('float')  #补充了+，考虑到同一视角下有同类别的多实例物体
                    instance_seg[:,:,category_id][mask.astype(bool)] = count_curr_instances_of_same_category[category]  #同一类别下的多实例编号从1开始
                    if category not in segmented_dict:
                        segmented_dict[category] = {}
                    
                    x_min,y_min,x_max,y_max = bbox[0]
                    if x_max == x_min:
                        x_max += 1
                    if y_max == y_min:
                        y_max += 1
                    adjust_bbox = (x_min, y_min, x_max, y_max)
                    normalized_bbox = torch.zeros_like(bbox, dtype=features.dtype)
                    normalized_bbox[0, 0] = bbox[0, 0] / width
                    normalized_bbox[0, 1] = bbox[0, 1] / height
                    normalized_bbox[0, 2] = bbox[0, 2] / width
                    normalized_bbox[0, 3] = bbox[0, 3] / height
            
                    ins_visual_feature = ops.roi_pool(features, [normalized_bbox], output_size=(1,1)).ravel()
                    ins_color_feature = self.get_color_feature(rgb, adjust_bbox)

                    concat_feature = torch.cat((ins_visual_feature, ins_color_feature)).ravel().cpu().detach().numpy()

                    instance_curr_id = str(category) + '_' + str(count_curr_instances_of_same_category[category])
                    instance_info = {
                        'score': obj['score'],
                        'mask': mask.astype(bool),
                        'feature': concat_feature,
                        'simulator_id': instance_curr_id
                    }
                    segmented_dict[category][instance_curr_id] = instance_info

        return instance_seg.astype(np.int8), segmented_dict

    def get_color_feature(self, rgb, bbox):
        # 转换为HSV颜色
        hsv = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2HSV)
        # 选取需要提取颜色直方图的区域
        x_min, y_min, x_max, y_max = bbox
        roi = hsv[y_min:y_max, x_min:x_max]
        # 分别计算色调值出现次数
        hist, bins = np.histogram(roi[:,:,0], bins=180, range=[0, 180])
        # 归一化
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        # 输出归一化后的颜色直方图向量
        # print(hist.shape)
        return torch.tensor(hist)





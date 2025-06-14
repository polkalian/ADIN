import typing, gym, torch, os, time, math
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, Union, List, cast, Sequence
from gym.spaces.dict import Dict as SpaceDict
from torchvision import utils as vutils
import pickle
from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
    DistributionType,
)
from allenact.base_abstractions.distributions import CategoricalDistr, DiagGaussian
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from allenact.utils.model_utils import make_cnn, compute_cnn_output
# from wbb_utils.cal_conf import cal_confounder
from torch.nn.parameter import Parameter
import pickle
from torchvision import models
import torchvision
# from roi_align import CropAndResize
import torch.nn.functional as F

from ivn_aff.mapper import Mapper
from ivn_aff.utils.utils import get_camera_matrix
# from ivn_aff.utils.segmentation import SegmentationHelper
import skimage
from ivn_aff.poni_model import *

from allenact_plugins.gym_plugin.gym_distributions import GaussianDistr

os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = "2"

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
action_space_num = {'decision': 4, 'nav': 5, 'pick': 5, 'move': 5}
fov = 90.0
map_size = 20
map_resolution = 0.05
STEP_SIZE = 0.25
channel = 128

# class LinearActorHead_(nn.Module):
#     def __init__(self, num_inputs: int, num_outputs: int):
#         super().__init__()

#         self.linear = nn.Linear(num_inputs, num_outputs)
#         nn.init.orthogonal_(self.linear.weight, gain=0.01)
#         nn.init.constant_(self.linear.bias, 0)

#     def forward(self, x: torch.FloatTensor):  # type: ignore
#         x = self.linear(x) # type:ignore

#         # noinspection PyArgumentList
#         return DiagGaussian(x)  # logits are [step, sampler, ...]


class RGBDSCNN(nn.Module):
    def __init__(
            self,
            observation_space: SpaceDict,
            output_size: int,
            layer_channels: Sequence[int] = (32, 64, 32),
            kernel_sizes: Sequence[Tuple[int, int]] = ((8, 8), (4, 4), (3, 3)),
            layers_stride: Sequence[Tuple[int, int]] = ((4, 4), (2, 2), (1, 1)),
            paddings: Sequence[Tuple[int, int]] = ((0, 0), (0, 0), (0, 0)),
            dilations: Sequence[Tuple[int, int]] = ((1, 1), (1, 1), (1, 1)),
            rgb_uuid: str = "rgb",
            depth_uuid: str = "depth",
            seg_uuid: str = "seg",
            flatten: bool = True,
            output_relu: bool = True,
    ):
        super().__init__()

        self.rgb_uuid = rgb_uuid
        if self.rgb_uuid in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces[self.rgb_uuid].shape[2]
            assert self._n_input_rgb >= 0
        else:
            self._n_input_rgb = 0

        self.depth_uuid = depth_uuid
        if self.depth_uuid in observation_space.spaces:
            self._n_input_depth = observation_space.spaces[self.depth_uuid].shape[2]
            assert self._n_input_depth >= 0
        else:
            self._n_input_depth = 0

        self.seg_uuid = seg_uuid
        self._n_input_seg = 1

        if not self.is_blind:
            # hyperparameters for layers
            self._cnn_layers_channels = list(layer_channels)
            self._cnn_layers_kernel_size = list(kernel_sizes)
            self._cnn_layers_stride = list(layers_stride)
            self._cnn_layers_paddings = list(paddings)
            self._cnn_layers_dilations = list(dilations)

            if self._n_input_rgb > 0:
                input_rgb_cnn_dims = np.array(
                    observation_space.spaces[self.rgb_uuid].shape[:2], dtype=np.float32
                )
                self.rgb_cnn = self.make_cnn_from_params(
                    output_size=output_size,
                    input_dims=input_rgb_cnn_dims,
                    input_channels=self._n_input_rgb,
                    flatten=flatten,
                    output_relu=output_relu,
                )

            if self._n_input_depth > 0:
                input_depth_cnn_dims = np.array(
                    observation_space.spaces[self.depth_uuid].shape[:2],
                    dtype=np.float32,
                )
                self.depth_cnn = self.make_cnn_from_params(
                    output_size=output_size,
                    input_dims=input_depth_cnn_dims,
                    input_channels=self._n_input_depth,
                    flatten=flatten,
                    output_relu=output_relu,
                )

            if self._n_input_seg > 0:
                input_seg_cnn_dims = np.array(
                    observation_space.spaces[self.depth_uuid].shape[:2],
                    dtype=np.float32,
                )
                self.seg_cnn = self.make_cnn_from_params(
                    output_size=output_size,
                    input_dims=input_seg_cnn_dims,
                    input_channels=self._n_input_depth,
                    flatten=flatten,
                    output_relu=output_relu,
                )

    def make_cnn_from_params(
            self,
            output_size: int,
            input_dims: np.ndarray,
            input_channels: int,
            flatten: bool,
            output_relu: bool,
    ) -> nn.Module:
        output_dims = input_dims
        for kernel_size, stride, padding, dilation in zip(
                self._cnn_layers_kernel_size,
                self._cnn_layers_stride,
                self._cnn_layers_paddings,
                self._cnn_layers_dilations,
        ):
            # noinspection PyUnboundLocalVariable
            output_dims = self._conv_output_dim(
                dimension=output_dims,
                padding=np.array(padding, dtype=np.float32),
                dilation=np.array(dilation, dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        # noinspection PyUnboundLocalVariable
        cnn = make_cnn(
            input_channels=input_channels,
            layer_channels=self._cnn_layers_channels,
            kernel_sizes=self._cnn_layers_kernel_size,
            strides=self._cnn_layers_stride,
            paddings=self._cnn_layers_paddings,
            dilations=self._cnn_layers_dilations,
            output_height=output_dims[0],
            output_width=output_dims[1],
            output_channels=output_size,
            flatten=flatten,
            output_relu=output_relu,
        )
        self.layer_init(cnn)

        return cnn

    @staticmethod
    def _conv_output_dim(
            dimension: Sequence[int],
            padding: Sequence[int],
            dilation: Sequence[int],
            kernel_size: Sequence[int],
            stride: Sequence[int],
    ) -> Tuple[int, ...]:
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                                (
                                        dimension[i]
                                        + 2 * padding[i]
                                        - dilation[i] * (kernel_size[i] - 1)
                                        - 1
                                )
                                / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    @staticmethod
    def layer_init(cnn) -> None:
        for layer in cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations: Dict[str, torch.Tensor]):  # type: ignore
        if self.is_blind:
            return None

        def check_use_agent(new_setting):
            if use_agent is not None:
                assert (
                        use_agent is new_setting
                ), "rgb and depth must both use an agent dim or none"
            return new_setting

        cnn_output_list: List[torch.Tensor] = []
        use_agent: Optional[bool] = None

        if self._n_input_rgb > 0:
            use_agent = check_use_agent(len(observations[self.rgb_uuid]) == 6)
            cnn_output_list.append(
                compute_cnn_output(self.rgb_cnn, observations[self.rgb_uuid])
            )

        if self._n_input_depth > 0:
            use_agent = check_use_agent(len(observations[self.depth_uuid]) == 6)
            cnn_output_list.append(
                compute_cnn_output(self.depth_cnn, observations[self.depth_uuid])
            )

        if self._n_input_seg > 0:
            use_agent = check_use_agent(len(observations[self.seg_uuid]) == 6)
            cnn_output_list.append(
                compute_cnn_output(self.seg_cnn, observations[self.seg_uuid])
            )

        if use_agent:
            channels_dim = 3  # [step, sampler, agent, channel (, height, width)]
        else:
            channels_dim = 2  # [step, sampler, channel (, height, width)]

        return torch.cat(cnn_output_list, dim=channels_dim)

class classfier(nn.Module):
    def __init__(self):
        super(classfier, self).__init__()
        self.linear = nn.Linear(514, 1)

    def forward(self, x):
        x = self.linear(x)
        return x

class ObstaclesNavRGBDActorCriticSimpleConvRNN(ActorCriticModel[GaussianDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Box,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        embed_coordinates=True,
        coordinate_embedding_dim=8,
        coordinate_dims=2,
        num_rnn_layers=1,
        rnn_type="GRU",
        mode='GT',
        action_std: float = 0.025,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.map_size = map_size

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True
            for p in self.sensor_fuser.parameters():
                p.requires_grad = False

            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size, "rgb", "depth")
        for p in self.visual_encoder.parameters():
            p.requires_grad = False

        # self.state_encoder = RNNStateEncoder(
        #     self._hidden_size,
        #     self._hidden_size,
        #     num_layers=num_rnn_layers,
        #     rnn_type=rnn_type,
        # )

        self.action_space = action_space
        self.mode = mode
        #主策略
        self.actor = nn.Sequential(
            nn.Linear(self._hidden_size, 2),
            nn.Sigmoid(),
        )
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )
            self.orientation_embedding = nn.Linear(
                3, coordinate_embedding_dim
            )

        # goal selection
        (   self.encoder,
            self.object_decoder,
            _
        ) = self.get_semantic_encoder_decoder(output_type="feature")
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        # for p in self.object_decoder.parameters():
        #     p.requires_grad = False

        self.linear = nn.Linear((0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size*2, self._hidden_size)
        
        if mode == 'yolov7':
            self.moveable_pred = nn.Linear(channel*7*7*2, 2)
            self.pickable_pred = nn.Linear(channel*7*7*2, 2)
            self.visible_pred = nn.Linear(channel*7*7*2, 2)
        
        # maximum standard deviation
        self.register_buffer(
            "action_std",
            torch.tensor([action_std] * 2).view(1, 1, -1),
            persistent=False,
        )

        self.data_num=0
        self.train()

    def get_semantic_encoder_decoder(self, output_type):
        num_categories = 8
        nsf = 32
        embedding_size = 64
        unet_bilinear_interp = True
        enable_area_head = True
        map_size = self.map_size/map_resolution

        encoder, object_decoder, area_decoder = None, None, None
        assert output_type in ["map", "dirs", "locs", "acts","feature"]
        encoder = UNetEncoder(
            num_categories,
            nsf,
            embedding_size,
            map_size
        )
        if output_type == "map":
            object_decoder = UNetDecoder(
                num_categories,
                nsf,
                bilinear=unet_bilinear_interp,
            )
        elif output_type == "dirs":
            object_decoder = DirectionDecoder(
                num_categories, ndirs, nsf
            )
        elif output_type == "locs":
            object_decoder = PositionDecoder(num_categories, nsf)
        elif output_type == "acts":
            object_decoder = ActionDecoder(
                num_categories, num_actions
            )
        elif output_type == "feature":
            object_decoder = FeatureDecoder(
                num_categories, nsf
            )
        if enable_area_head:
            area_decoder = UNetDecoder(
                1,
                nsf,
                bilinear=unet_bilinear_interp,
            )

        return encoder, object_decoder, area_decoder
    
    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    # @property
    # def num_recurrent_layers(self):
    #     return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            goal = observations['target_coordinates_ind'] / map_resolution
            return self.coordinate_embedding(
                goal.to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)
        
    def get_orientation_encoding(self, observations):
        if self.embed_coordinates:
            pose = observations['maps']['pose_pred'][:,:,:3]
            return self.orientation_embedding(
                pose.to(torch.float32)
            )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        # return dict(
        #     rnn=(
        #         (
        #             ("layer", self.num_recurrent_layers),
        #             ("sampler", None),
        #             ("hidden", self.recurrent_hidden_state_size),
        #         ),
        #         torch.float32,
        #     ),
        # )
        return None

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,  #步数step
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        # print(observations['maps'])
        target_encoding = self.get_target_coordinates_encoding(observations)
        orientation_encoding = self.get_orientation_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding] + [orientation_encoding]

        if True:
            observations['rgb'] = torch.where(torch.isinf(observations['rgb']), torch.full_like(observations['rgb'], 1),
                                              observations['rgb'])
            observations['depth'] = torch.where(torch.isinf(observations['depth']),
                                                torch.full_like(observations['depth'], 1),
                                                observations['depth'])
            observations['rgb'] = torch.where(torch.isnan(observations['rgb']), torch.full_like(observations['rgb'], 1),
                                              observations['rgb'])
            observations['depth'] = torch.where(torch.isnan(observations['depth']),
                                                torch.full_like(observations['depth'], 1),
                                                observations['depth'])

        # with torch.cuda.device('cuda:1'):
        #     torch.cuda.empty_cache()
        # print(observations['maps']['feature'])
        
        # with torch.no_grad():
        if observations['maps']['feature'].dim() == 2: # GT
            semantic_map = observations['maps']['semantic'][:,:,:,:,:8]
        else:
            semantic_map = observations['maps']['semantic'][:,:,:,:,:15] # yolov7
            semantic_map_= observations['maps']['semantic'][:,:,:,:,:8]
            for step in range(semantic_map.shape[0]):
                feature_sample = []
                for sample in range(semantic_map.shape[1]):
                    features = observations['maps']['feature'][step,sample,:,:]
                    moveable = torch.softmax(self.moveable_pred(features),dim=-1)[:,0]
                    semantic_map_[step,sample,:,:,5] = torch.clip(sum([semantic_map[step,sample,:,:,i+5]*moveable[i] for i in range(moveable.shape[0])]),0,1)
                    pickable = torch.softmax(self.pickable_pred(features),dim=-1)[:,0]
                    semantic_map_[step,sample,:,:,6] = torch.clip(sum([semantic_map[step,sample,:,:,i+5]*pickable[i] for i in range(pickable.shape[0])]),0,1)
                    visible = torch.softmax(self.visible_pred(features),dim=-1)[:,0]
                    semantic_map_[step,sample,:,:,7] = torch.clip(sum([semantic_map[step,sample,:,:,i+5]*visible[i] for i in range(visible.shape[0])]),0,1)
            semantic_map = semantic_map_

        # semantic_map = np.swapaxes(semantic_map[0,:,:,:], 3, 1).float()
        semantic_map = np.swapaxes(semantic_map, 4, 2).float()
        semantic_map = np.swapaxes(semantic_map, 3, 4)
        # # 多步多采样器
        feature_step = []
        for step in range(semantic_map.shape[0]):
            feature_sample = []
            for sample in range(semantic_map.shape[1]):
                map_embedding = self.encoder(semantic_map[step,sample,:,:,:].unsqueeze(0))
                feature = self.object_decoder(map_embedding)
                feature_sample.append(feature.unsqueeze(0))
            feature = torch.hstack(tuple(feature_sample))
            feature_step.append(feature)
        feature = torch.vstack(tuple(feature_step))

        # 预测可达性地图
        # feature_step = []
        # for step in range(semantic_map.shape[0]):
        #     feature_sample = []
        #     for sample in range(semantic_map.shape[1]):
        #         map_embedding = self.encoder(semantic_map[step,sample,:,:,:].unsqueeze(0))
        #         affordance_map = torch.softmax(self.affordance_pred(map_embedding), dim=1)
        #         semantic_map[step,sample,4,:,:] = affordance_map[0,0,:,:]
        #         map_embedding = self.encoder(semantic_map[step,sample,:,:,:].unsqueeze(0))
        #         feature = self.object_decoder(map_embedding)
        #         feature_sample.append(feature.unsqueeze(0))
        #     feature = torch.hstack(tuple(feature_sample))
        #     feature_step.append(feature)
        # feature = torch.vstack(tuple(feature_step))

        x = [feature] + x

        x = torch.cat(x, dim=-1)

        x = nn.Sigmoid()(self.linear(x))

        # x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)
        action = self.actor(x)

        # if x.shape[0] == 30:
        #     print(torch.cuda.max_memory_cached(device='cuda:1'))
        #     print(torch.cuda.memory_cached(device='cuda:1')
        
        ac_output = ActorCriticOutput(
            distributions=cast(DistributionType, GaussianDistr(loc=action, scale=self.action_std)),
            values=self.critic(x), extras={"Aff": {'mode': 'aux'}}
        )

        # memory.set_tensor("rnn", rnn_hidden_states)

        return ac_output, None


class ObstaclesNavRGBDSActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 3, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = RGBDSCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class ObstaclesNavRGBDKNIEActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size, "rgb", "depth")

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size + obstacle_state_hidden_dim * action_space.n,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NIE
        self.NIE = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NIE[4].weight.data.zero_()
        self.NIE[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NIE attention
        self.NIE_atten = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 1)
        )

        # NIE Summary
        self.NIE_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 4, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)

        hidden_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NIE_hidden = self.NIE(hidden_feature)
        NIE_hidden = NIE_hidden
        M = NIE_hidden.view(nb, ng, no, na, 3, 4)
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NIE_atten_score = self.NIE_atten(atten_feature)
        NIE_atten_prob = nn.functional.softmax(NIE_atten_score, 2)
        NIE_atten_hidden = (hidden_feature * NIE_atten_prob).sum(2)
        NIE_atten_hidden = self.NIE_summary(NIE_atten_hidden)
        NIE_atten_hidden = NIE_atten_hidden.view(nb, ng, -1)
        x.append(NIE_atten_hidden)

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x),
            values=self.critic(x),
            extras={"nie_output": new_keypoints}
        )

        return out, memory.set_tensor("rnn", rnn_hidden_states)


class ObstaclesNavRGBDKvNIEActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size, "rgb", "depth")

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.state_encoder_ = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size + obstacle_state_hidden_dim * action_space.n,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor = LinearActorHead(self._hidden_size, 10)
        self.critic = LinearCriticHead(self._hidden_size)

        self.actor_ = LinearActorHead(self._hidden_size, 12)
        self.critic_ = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NIE
        self.NIE = nn.Sequential(
            nn.Linear(hidden_size + self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NIE[4].weight.data.zero_()
        self.NIE[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NIE attention
        self.NIE_atten = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 1)
        )

        # NIE Summary
        self.NIE_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 4, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        time1 = time.time()
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        observations['rgb'] = torch.where(torch.isinf(observations['rgb']), torch.full_like(observations['rgb'], 1),
                                          observations['rgb'])
        observations['depth'] = torch.where(torch.isinf(observations['depth']),
                                            torch.full_like(observations['depth'], 1),
                                            observations['depth'])
        observations['rgb'] = torch.where(torch.isnan(observations['rgb']), torch.full_like(observations['rgb'], 1),
                                          observations['rgb'])
        observations['depth'] = torch.where(torch.isnan(observations['depth']),
                                            torch.full_like(observations['depth'], 1),
                                            observations['depth'])
        observations['3Dkeypoints_local'] = torch.where(torch.isnan(observations['3Dkeypoints_local']),
                                            torch.full_like(observations['3Dkeypoints_local'], 1),
                                            observations['3Dkeypoints_local'])
        observations['3Dkeypoints_local'] = torch.where(torch.isinf(observations['3Dkeypoints_local']),
                                                        torch.full_like(observations['3Dkeypoints_local'], 1),
                                                        observations['3Dkeypoints_local'])


        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)
        perception_embed_hidden = perception_embed.view(nb, ng, 1, 1, self._hidden_size).repeat(1, 1, no, na, 1)

        hidden_feature = torch.cat((perception_embed_hidden, obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NIE_hidden = self.NIE(hidden_feature)
        NIE_hidden = NIE_hidden
        M = NIE_hidden.view(nb, ng, no, na, 3, 4)
        # M_test = M.clone()
        # M_test[:, :, :, 7] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        # M_test[:, :, :, 5] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NIE_atten_score = self.NIE_atten(atten_feature)
        NIE_atten_prob = nn.functional.softmax(NIE_atten_score, 2)
        NIE_atten_hidden = (hidden_feature * NIE_atten_prob).sum(2)
        NIE_atten_hidden = self.NIE_summary(NIE_atten_hidden)
        NIE_atten_hidden = NIE_atten_hidden.view(nb, ng, -1)
        x.append(NIE_atten_hidden)

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder_(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor_(x),
            values=self.critic_(x),
            extras={"nie_output": new_keypoints}
        )

        return out, memory.set_tensor("rnn", rnn_hidden_states)


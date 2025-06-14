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
from allenact.base_abstractions.distributions import CategoricalDistr
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

from ivn_proc.intervention_classifier import Interventional_Classifier

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
action_space_num = {'decision': 4, 'nav': 5, 'pick': 5, 'move': 5}


class LinearActorHead_(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.FloatTensor):  # type: ignore
        x = self.linear(x) # type:ignore

        # noinspection PyArgumentList
        return x  # logits are [step, sampler, ...]


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

class ObstaclesNavRGBDActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
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
        mode='nav',
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.intent_size = 12
        self.intent_embedding_size = 12
        self.embed_intent = True

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

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.state_encoder_ = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        # for p in self.state_encoder.parameters():
        #     p.requires_grad = False

        self.state_encoder_intent = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size + self.intent_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        for p in self.state_encoder_intent.parameters():
            p.requires_grad = False

        self.action_space = action_space
        self.mode = mode
        #主策略
        self.actor = LinearActorHead(self._hidden_size, 10)
        self.critic = LinearCriticHead(self._hidden_size)

        self.actor_ = LinearActorHead(self._hidden_size, 12)
        self.critic_ = LinearCriticHead(self._hidden_size)

        self.actor_intent = LinearActorHead(self._hidden_size, 12)
        for p in self.actor_intent.parameters():
            p.requires_grad = False

        # self.actor_intent_15 = LinearActorHead(self._hidden_size, 12)
        # for p in self.actor_intent_15.parameters():
        #     p.requires_grad = False

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )
        if self.embed_intent:
            self.intent_embedding = nn.Linear(
                self.intent_size, self.intent_embedding_size
            )
        if self.embed_intent:
            self.intent_embedding_ = nn.Linear(
                self.intent_size, self.intent_embedding_size
            )
        for p in self.intent_embedding_.parameters():
            p.requires_grad = False

        self.data_num=0
        self.train()
        self.mode = [0]*40
        self.last_step_size = 1
        self.count = 0

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
            ),
            rnn_=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
            # rnn_p=(
            #     (
            #         ("layer", self.num_recurrent_layers),
            #         ("sampler", None),
            #         ("hidden", self.recurrent_hidden_state_size),
            #     ),
            #     torch.float32,
            # ),
            rnn_intent=(
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
        masks: torch.FloatTensor,  #步数step
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]
        x_ = x
        x_p=x

        # 每个epoch更新意图的参数
        if self.last_step_size == 30:
            self.count += 1
        # if self.count%12 == 0 and self.count != 0 and self.last_step_size == 30:
        # if self.count % 15 == 0 and self.count != 0 and self.last_step_size == 30:
        #     self.state_encoder_intent.load_state_dict(self.state_encoder_.state_dict())
        #     self.actor_intent.load_state_dict(self.actor_.state_dict())
        #     self.intent_embedding_.load_state_dict(self.intent_embedding.state_dict())
        # if self.count % 15 == 0 and self.count != 0 and self.last_step_size == 30:
        #     self.actor_intent_15.load_state_dict(self.actor_.state_dict())


            # for name, parms in self.state_encoder_intent.named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_value:', parms.grad)
            #     print("===")
            # for name, parms in self.state_encoder.named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_value:', parms.grad)
            #     print("===")

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

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)

            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)

        #ppo 不加意图
        device_ = x.device
        void_intent = torch.full((x.shape[0],x.shape[1],self.intent_size), 1/12).to(device_)

        if self.embed_intent:
            void_intent = self.intent_embedding_(void_intent)

        x_intent = torch.cat((void_intent,x), dim=2)

        x_intent, rnn_hidden_states_intent = self.state_encoder_intent(x_intent, memory.tensor("rnn_intent"), masks)
        intent = self.actor_intent(x_intent)
        intent = torch.softmax(intent.logits, dim=-1)
        # 15 intent
        # intent_15 = self.actor_intent_15(x_intent)
        # intent_15 = torch.softmax(intent_15.logits, dim=-1)
        # intent_save = intent_15
        # print(intent)
        if self.embed_intent:
            intent = self.intent_embedding(intent)

        # x = torch.cat((intent, x), dim=2)
        x, rnn_hidden_states = self.state_encoder_(x, memory.tensor("rnn_"), masks)

        # try:
        #     memory.tensor('rnn_')
        # except:
        #     memory.check_append('rnn_', memory.tensor('rnn'), memory.sampler_dim('rnn'))
        # if not self.is_blind:
        #     perception_embed_ = self.visual_encoder_(observations)
        #     #perception_embed_d = self.visual_encoder_(observations)
        #     if self.sensor_fusion:
        #         perception_embed_ = self.sensor_fuser_(perception_embed_)
        #
        #     x_=[perception_embed_] + x_
        #
        # x_ = torch.cat(x_, dim=-1)
        #
        # x_, rnn_hidden_states_ = self.state_encoder_(x_, memory.tensor("rnn_"), masks)
        #
        # #pick 模块
        # if not self.is_blind:
        #     perception_embed_p = self.visual_encoder_pick(observations)
        #     if self.sensor_fusion:
        #         perception_embed_p = self.sensor_fuser_pick(perception_embed_p)
        #
        #     x_p = [perception_embed_p] + x_p
        #
        # x_p = torch.cat(x_p, dim=-1)
        # x_p, rnn_hidden_states_p = self.state_encoder_pick(x_p, memory.tensor("rnn_p"), masks)
        #
        #
        #
        #
        # # 推动模块训练
        # intent = torch.softmax(self.actor_2(x), dim=-1)
        # x = torch.cat((intent, x_intent), 2)
        # x, rnn_hidden_states_intnet = self.state_encoder_intent(x, memory.tensor("rnn_intent"), masks)
        # cat = self.actor_2_(x)
        #
        # cat_1 = cat[:,:,:4]
        # cat_2= cat[:,:,-1:]
        # critic = self.critic_2_(x)
        #
        # action = torch.zeros([x.shape[0], x.shape[1], 5]).to(device_)
        # action += float("-inf")
        # action_ = torch.zeros([x.shape[0], x.shape[1], 2]).to(device_)
        # action_ += float("-inf")
        #
        # action = torch.cat((action, cat_1), 2)
        # action = torch.cat((action, action_), 2)
        # action = torch.cat((action, cat_2), 2)
        # print(action)

        # 导航模块训练
        # cat = self.actor_1_(x_)
        # critic = self.critic_1_(x_)
        
        # action = torch.zeros([x.shape[0], x.shape[1], 6]).to(device_)
        # action += float("-inf")
        
        # action = torch.cat((cat[:,:,:5], action), 2)
        # action = torch.cat((action, cat[:,:,-1:]), 2)
        # data={}
        # data['feature']=perception_embed_
        # data['Q']=critic
        # with open('datasets/conf_nav/data' + str(self.data_num) + '.pickle', 'wb')as t:
        #     pickle.dump(data, t)
        # self.data_num += 1

        # 拾起模块训练
        # cat = self.actor_3(x_p)
        # cat_1 = cat[:, :, :2]
        # cat_2 = cat[:, :, -3:]
        # critic = self.critic_3(x_p)
        #
        # action_1 = torch.zeros([x.shape[0], x.shape[1], 1]).to(device_)
        # action_1 += float("-inf")
        # action_2 = torch.zeros([x.shape[0], x.shape[1], 6]).to(device_)
        # action_2 += float("-inf")
        #
        # action = torch.cat((action_1, cat_1), 2)
        # action = torch.cat((action, action_2), 2)
        # action = torch.cat((action, cat_2), 2)
        # print(action)

        # 决策模块训练
        # cat = self.actor(x_d).logits
        # #decision = torch.argmax(cat, dim=-1)
        # critic = self.critic(x_d)
        # action = torch.zeros([x.shape[0], x.shape[1], self.action_space.n]).to(device_)
        # action += float("-inf")
        #
        # for step in range(cat.shape[0]):
        #     for sample in range(cat.shape[1]):
        #         # nav
        #         ind = observations['low_nav'][step][sample].cpu().numpy().tolist()[0]
        #         index = (torch.LongTensor([step]),
        #                  torch.LongTensor([sample]),
        #                  torch.LongTensor([ind]))
        #         action.index_put_(index, cat[step][sample][0])
        #
        #         # move
        #         ind = observations['low_move'][step][sample].cpu().numpy().tolist()[0]
        #         index = (torch.LongTensor([step]),
        #                  torch.LongTensor([sample]),
        #                  torch.LongTensor([ind]))
        #         if action[step][sample][ind] < cat[step][sample][1]:
        #             action.index_put_(index, cat[step][sample][1])  # 4维的值
        #
        #         # pick
        #         ind = observations['low_pick'][step][sample].cpu().numpy().tolist()[0]
        #         index = (torch.LongTensor([step]),
        #                  torch.LongTensor([sample]),
        #                  torch.LongTensor([ind]))
        #         if action[step][sample][ind] < cat[step][sample][2]:
        #             action.index_put_(index, cat[step][sample][2])
        #
        #         # done
        #         index = (torch.LongTensor([step]),
        #                  torch.LongTensor([sample]),
        #                  torch.LongTensor([11]))
        #         action.index_put_(index, cat[step][sample][3])

        # # 整体训练
        # #data={}
        # cat = self.actor(x_).logits
        # critic = self.critic(x_)
        # decision = torch.softmax(cat, dim=-1)
        #
        #
        #
        #
        # # #测试时使用
        # # # # data['feature']=perception_embed_
        # # # # data['Q']=critic
        # # # with open('conf_ours/data' + str(self.data_num) + '.pickle', 'wb')as t:
        # # #     pickle.dump(data, t)
        # # # self.data_num += 1
        # # #print('critic:',critic.shape)
        # # # print(decision)
        #
        # action = torch.zeros([x.shape[0], x.shape[1], self.action_space.n]).to(device_)
        # # action += float("-inf")
        # for sample in range(x.shape[1]):
        #     for step in range(x.shape[0]):
        #         # print(prev_actions[0][sample])
        #         #存储数据完毕
        #         if prev_actions[0][sample] in [5,6,7,8]:
        #             self.mode[sample] = 1
        #         if prev_actions[0][sample] in [9,10]:
        #             self.mode[sample] = 2
        #         if self.mode[sample] != 0 and prev_actions[0][sample] == 11:
        #             self.mode[sample] = 0
        #
        #         ind = [i+5 for i in range(action_space_num['move']-1)]
        #         ind.extend([j for j in range(11, 12)])
        #         index = (torch.LongTensor([step for i in range(action_space_num['move'])]).to(device_),
        #                  torch.LongTensor([sample for i in range(action_space_num['move'])]).to(device_),
        #                  torch.LongTensor(ind).to(device_))
        #         # move
        #         # ind = observations['low_move'][step][sample].cpu().numpy().tolist()[0]
        #         # index = (torch.LongTensor([step]),
        #         #          torch.LongTensor([sample]),
        #         #          torch.LongTensor([ind]))
        #         actor_2 = self.actor_2(x[step][sample].view(1, 1, -1)).view(-1)
        #         if self.mode[sample] == 0:
        #             action.index_put_(index, torch.softmax(actor_2, dim=-1) * decision[step][sample][1])
        #         elif self.mode[sample] == 1:
        #             action.index_put_(index, torch.softmax(actor_2, dim=-1))
        #         # if self.mode[sample] == 2:
        #         #     action.index_put_(index, torch.softmax(actor_2, dim=-1)*0)
        #
        #         ind = [i for i in range(1, 3)]
        #         ind.extend([j for j in range(9, 11)])
        #         ind.extend([j for j in range(11, 12)])
        #         index = (torch.LongTensor([step for i in range(action_space_num['pick'])]).to(device_),
        #                  torch.LongTensor([sample for i in range(action_space_num['pick'])]).to(device_),
        #                  torch.LongTensor(ind).to(device_))
        #         # pick
        #         # ind = observations['low_pick'][step][sample].cpu().numpy().tolist()[0]
        #         # index = (torch.LongTensor([step]),
        #         #          torch.LongTensor([sample]),
        #         #          torch.LongTensor([ind]))
        #         actor_3 = self.actor_3(x_p[step][sample].view(1, 1, -1)).view(-1)
        #         if self.mode[sample] == 0:
        #             action.index_put_(index, torch.softmax(actor_3, dim=-1) * decision[step][sample][2])
        #         elif self.mode[sample] == 2:
        #             action.index_put_(index, torch.softmax(actor_3, dim=-1))
        #         # if self.mode[sample] == 1:
        #         #     ind = [i for i in range(1, 3)]
        #         #     ind.extend([j for j in range(9, 11)])
        #         #     index = (torch.LongTensor([step for i in range(action_space_num['pick']-1)]).to(device_),
        #         #              torch.LongTensor([sample for i in range(action_space_num['pick']-1)]).to(device_),
        #         #              torch.LongTensor(ind).to(device_))
        #         #     action.index_put_(index, torch.zeros(4).to(device_)*0)
        #
        #         if self.mode[sample] == 0:
        #             # index = (torch.LongTensor([step]).to(device_),
        #             #          torch.LongTensor([sample]).to(device_),
        #             #          torch.LongTensor([11]).to(device_))
        #             # temp = torch.zeros([1]).to(device_)
        #             # action.index_put_(index, temp)
        #
        #             ind = [i for i in range(action_space_num['nav'])]
        #             index = (torch.LongTensor([step for i in range(action_space_num['nav'])]).to(device_),
        #                      torch.LongTensor([sample for i in range(action_space_num['nav'])]).to(device_),
        #                      torch.LongTensor(ind).to(device_))
        #             # index_ = (torch.LongTensor([step]).to(device_), torch.LongTensor([sample]).to(device_), torch.LongTensor([0]).to(device_))
        #             # nav
        #             # ind = observations['low_nav'][step][sample].cpu().numpy().tolist()[0]
        #             # index = (torch.LongTensor([step]),
        #             #          torch.LongTensor([sample]),
        #             #          torch.LongTensor([ind]))
        #             actor_1 = self.actor_1_(x_[step][sample].view(1, 1, -1))[:,:,:5].view(-1)
        #             # action.index_put_(index, decision[step][sample][0])
        #             action.index_put_(index, torch.softmax(actor_1, dim=-1) * decision[step][sample][0])
        #
        #             # END
        #             index = (torch.LongTensor([step]).to(device_),
        #                      torch.LongTensor([sample]).to(device_),
        #                      torch.LongTensor([self.action_space.n - 1]).to(device_))
        #             action.index_put_(index, decision[step][sample][3])

        # if self.mode[sample] != 0:
        # for name, parms in self.sensor_fuser_.named_parameters():
        #     print('-->name:', name)
        #     print('-->para:', parms)
        #     print('-->grad_requirs:', parms.requires_grad)
        #     print('-->grad_value:', parms.grad)
        #     print("===")

        # action = CategoricalDistr(logits=action)
        # action = CategoricalDistr(probs=intent_save)

        try:
            ac_output = ActorCriticOutput(
                distributions=self.actor_(x), values=self.critic_(x), extras={}
            )
            # ac_output = ActorCriticOutput(
            #     distributions=action, values=self.critic_(x), extras={'intent': intent_save}
            # )
            # print(torch.softmax(self.actor(x).logits, dim=-1))
        except:
            print('rgb',torch.isnan(observations['rgb']).any(),torch.isinf(observations['rgb']).any())
            print('depth',torch.isnan(observations['depth']).any(),torch.isinf(observations['depth']).any())

        memory.set_tensor("rnn_", rnn_hidden_states)
        # memory.set_tensor("rnn_", rnn_hidden_states_)
        # memory.set_tensor("rnn_p", rnn_hidden_states_p)
        memory.set_tensor("rnn_intent", rnn_hidden_states_intent)

        self.last_step_size = x.shape[0]

        return ac_output, memory


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


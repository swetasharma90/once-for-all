import random

from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicConvLayer,
    DynamicLinearLayer,
    DynamicInceptionBlock,
    # Add DynamicInceptionBlock
)
'''
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicResNetBottleneckBlock,
)
'''
from ofa.utils.layers import IdentityLayer, InceptionBlock
from ofa.imagenet_classification.networks import Inception
from ofa.utils import make_divisible, val2list, MyNetwork

__all__ = ["OFAInception"]


class OFAInception(Inception):
    def __init__(
        self,
        n_classes=1000,    # For image-net
        bn_param=(0.1, 1e-5), # Check if you really need this
        dropout_rate=0.4,
        depth_list= [0, 1],
        width_mult_list=[0.5, 1.0, 2.0],
    ):
        self.dropout_rate = dropout_rate
        #print('depth list',depth_list)
        self.depth_list = val2list(depth_list)
        self.width_mult_list = val2list(width_mult_list)
        # sort
        self.depth_list.sort() # Ascending
        self.width_mult_list.sort() # Ascending

        input_channel = [
            make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            for width_mult in self.width_mult_list
        ]
        mid_input_channel = [
            make_divisible(channel * 3, MyNetwork.CHANNEL_DIVISIBLE)
            for channel in input_channel
        ]

        stage_width_list = Inception.STAGE_WIDTH_LIST.copy()
        stage_width_sub_list = Inception.STAGE_WIDTH_SUB_LIST.copy()

        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = [  # For each block a list of possible widths
                make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
                for width_mult in self.width_mult_list
            ]

        for i, width in enumerate(stage_width_sub_list):
            stage_width_sub_list[i] = [ [# For each block a list of possible sub widths
                make_divisible(sub_width * width_mult, MyNetwork.CHANNEL_DIVISIBLE) for sub_width in width]
                for width_mult in self.width_mult_list
            ]
        #print('base depth',Inception.BASE_DEPTH_LIST)
        #print('list depth', self.depth_list)
        n_block_list = [  # For each block
            base_depth + max(self.depth_list) for base_depth in Inception.BASE_DEPTH_LIST
        ]
        stride_list = [1, 1, 1, 1, 1, 1, 1, 1, 1] # stride for each block

        #print(input_channel)
        # build input stem
        input_stem1 = [
            DynamicConvLayer(
                [3],
                input_channel,
                kernel_size_list=[7],
                stride=2,
                use_bn=True,
                act_func="relu",
                #ops_order="weight_bn_act",  # For now use Batch Normalization but test without batch norm as well.
            )
        ]

        input_stem2 = [
            DynamicConvLayer(
                input_channel,
                mid_input_channel,
                kernel_size_list=[3],
                stride=1,
                use_bn=True,
                act_func="relu",
                #ops_order="weight_bn_act",  # For now use Batch Normalization but test without batch norm as well.
            )
        ]

        # blocks
        blocks = []
        #print('block list',n_block_list)
        #print(zip(n_block_list, stage_width_list, stage_width_sub_list, stride_list))
        for d, width, sub_width, s in zip(n_block_list, stage_width_list, stage_width_sub_list, stride_list):
            #print('depth of each block \n',d)
            #print('width of each block \n', width)
            #print('depth of each block \n', d)
            subblock = []
            for i in range(d):
                #stride = s if i == 0 else 1
                inception_block = DynamicInceptionBlock(  # Write this dynamic block
                    mid_input_channel,
                    width,
                    sub_width,
                    stride=s,
                    act_func="relu",
                )
                subblock.append(inception_block)
                mid_input_channel = width
            subblock.append(IdentityLayer(width, width))  # Add Identity layer
            blocks.append(subblock)

        # classifier
        classifier = DynamicLinearLayer(
            mid_input_channel, n_classes  # Remove this dropout
        )

        super(OFAInception, self).__init__([input_stem1, input_stem2], blocks, dropout_rate = dropout_rate, classifier = classifier)

        # set bn param
        self.set_bn_param(*bn_param)

        # runtime_depth
        self.input_stem_skipping = 0   # Check what this is doing
        self.runtime_depth = [0] * len(n_block_list)  # For now it's 0 but changes during execution for each block

    @property
    def ks_list(self):  # Not getting used anywhere
        return [3]

    @staticmethod
    def name():
        return "OFAInception"

    @property
    def grouped_block_index(self):  # Check the role of Identity and modify this function accordigly
        info_list = []  # Identity layer acts as a marker between two stages and maintains indexes to
        block_index_list = []  # individual blocks within each stage
        # Used for only DYNAMIC INCEPTION Networks

        for i, block in enumerate(self.blocks3):
            if (
                    isinstance(block.module, IdentityLayer)
                    and len(block_index_list) > 0
            ):
                info_list.append(block_index_list)
                block_index_list = []
            else:
                block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        for i, block in enumerate(self.blocks4):
            if (
                    isinstance(block.module, IdentityLayer)
                    and len(block_index_list) > 0
            ):
                info_list.append(block_index_list)
                block_index_list = []
            else:
                block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        for i, block in enumerate(self.blocks5):
            if (
                    isinstance(block.module, IdentityLayer)
                    and len(block_index_list) > 0
            ):
                info_list.append(block_index_list)
                block_index_list = []
            else:
                block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    def forward(self, x):
        #from pprint import pprint
        #pprint(vars(self))
        #print("grouped",super().grouped_block_index)
        #print("x size",x.size())
        for layer in self.input_stem1:
            x = layer(x)
        x = self.max_pooling1(x)
        #print("x size", x.size())
        for layer in self.input_stem2:
            x = layer(x)
        x = self.max_pooling2(x)
        #print("x size", x.size())
        #print(super().grouped_block_index)
        for stage_id, block_idx in enumerate(super().grouped_block_index): #(self.grouped_block_index()):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            if(stage_id in [0,1]):
                for idx in active_idx:
                    #print('blocks3   ',idx,x.size())
                    #print(self.blocks3[idx])
                    x = self.blocks3[idx](x)
            if (stage_id in [2, 3, 4, 5, 6]):
                for idx in active_idx:
                    #print('blocks4\n\n')
                    x = self.blocks4[idx](x)
            if (stage_id in [7, 8]):
                for idx in active_idx:
                    #print('blocks5\n\n')
                    x = self.blocks5[idx](x)
            if(stage_id == 1):
                x = self.max_pooling3(x)
            if (stage_id == 6):
                x = self.max_pooling4(x)

        #print('before pool',x.size())
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ""
        for layer in self.input_stem1:
            _str += layer.module_str + "\n"
        _str += "max_pooling(ks=3, stride=2)\n"
        for layer in self.input_stem2:
            _str += layer.module_str + "\n"
        _str += "max_pooling(ks=3, stride=2)\n"
        for stage_id, block_idx in enumerate(super().grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                if (stage_id in [0, 1]):
                    for idx in active_idx:
                        #print(self.blocks3[idx] )
                        _str += self.blocks3[idx].module_str + "\n"
                if (stage_id in [2, 3, 4, 5, 6]):
                    for idx in active_idx:
                        _str += self.blocks4[idx].module_str + "\n"
                if (stage_id in [7, 8]):
                    for idx in active_idx:
                        _str += self.blocks5[idx].module_str + "\n"

            if (stage_id == 0):
                _str += "max_pooling(ks=3, stride=2)\n"
            if (stage_id == 6):
                _str += "max_pooling(ks=3, stride=2)\n" "\n"
        _str += self.avg_pool.__repr__() + "\n"
        if(self.dropout_rate>0):
            _str += self.dropout.__repr__() + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": OFAInception.__name__,
            "bn": self.get_bn_param(),
            "input_stem1": [layer.config for layer in self.input_stem1],
            "input_stem2": [layer.config for layer in self.input_stem2],
            "blocks3": [block.config for block in self.blocks3],
            "blocks4": [block.config for block in self.blocks4],
            "blocks5": [block.config for block in self.blocks5],
            "dropout": self.dropout_rate,
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError("do not support this function")

    def load_state_dict(self, state_dict, **kwargs): # Check if this needs any change
        model_dict = self.state_dict()
        #print('model dict\n',model_dict)
        #print('load dict')
        for key in state_dict:
            new_key = key
            #print(new_key)
            if new_key in model_dict:
                pass
            elif ".linear." in new_key:
                new_key = new_key.replace(".linear.", ".linear.linear.")
            elif "bn." in new_key:
                new_key = new_key.replace("bn.", "bn.bn.")
            elif "conv.weight" in new_key:
                new_key = new_key.replace("conv.weight", "conv.conv.weight")
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, "%s" % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAInception, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_max_net(self):
        self.set_active_subnet(
            d=max(self.depth_list),
            w=len(self.width_mult_list) - 1,
        )

    def set_active_subnet(self, d=None, w=None, **kwargs):
        depth = val2list(d, len(Inception.BASE_DEPTH_LIST)) # What is +1 for? --> It was for Resnet
        width_mult = val2list(w, len(Inception.BASE_DEPTH_LIST) + 2)

        if width_mult[0] is not None:
            self.input_stem1[0].active_out_channel = self.input_stem1[0].out_channel_list[width_mult[0]]
        if width_mult[1] is not None:
            self.input_stem2[0].active_out_channel = self.input_stem2[0].out_channel_list[width_mult[1]]

        for stage_id, (block_idx, d, w) in enumerate(
            zip(super().grouped_block_index, depth[0:], width_mult[2:])
        ):
            #print('setting active subnet',stage_id, (block_idx, d, w))
            if d is not None:
                self.runtime_depth[stage_id] = max(self.depth_list) - d
            if w is not None:
                for idx in block_idx:
                    if (stage_id in [0, 1]):
                        self.blocks3[idx].active_out_channel = self.blocks3[
                                idx
                            ].out_channel_list[w]
                        self.blocks3[idx].active_ch1x1_out_channel = self.blocks3[
                            idx
                        ].ch1x1_out_channel_list[w]
                        self.blocks3[idx].active_ch3x3red_out_channel = self.blocks3[
                            idx
                        ].ch3x3red_out_channel_list[w]
                        self.blocks3[idx].active_ch3x3_out_channel = self.blocks3[
                            idx
                        ].ch3x3_out_channel_list[w]
                        self.blocks3[idx].active_ch5x5red_out_channel = self.blocks3[
                            idx
                        ].ch5x5red_out_channel_list[w]
                        self.blocks3[idx].active_ch5x5_out_channel = self.blocks3[
                            idx
                        ].ch5x5_out_channel_list[w]
                        self.blocks3[idx].active_pool_proj_out_channel = self.blocks3[
                            idx
                        ].pool_proj_out_channel_list[w]
                    if (stage_id in [2, 3, 4, 5, 6]):
                        self.blocks4[idx].active_out_channel = self.blocks4[
                                idx
                            ].out_channel_list[w]
                        self.blocks4[idx].active_ch1x1_out_channel = self.blocks4[
                            idx
                        ].ch1x1_out_channel_list[w]
                        self.blocks4[idx].active_ch3x3red_out_channel = self.blocks4[
                            idx
                        ].ch3x3red_out_channel_list[w]
                        self.blocks4[idx].active_ch3x3_out_channel = self.blocks4[
                            idx
                        ].ch3x3_out_channel_list[w]
                        self.blocks4[idx].active_ch5x5red_out_channel = self.blocks4[
                            idx
                        ].ch5x5red_out_channel_list[w]
                        self.blocks4[idx].active_ch5x5_out_channel = self.blocks4[
                            idx
                        ].ch5x5_out_channel_list[w]
                        self.blocks4[idx].active_pool_proj_out_channel = self.blocks4[
                            idx
                        ].pool_proj_out_channel_list[w]
                    if (stage_id in [7, 8]):
                        self.blocks5[idx].active_out_channel = self.blocks5[
                                idx
                            ].out_channel_list[w]
                        self.blocks5[idx].active_ch1x1_out_channel = self.blocks5[
                            idx
                        ].ch1x1_out_channel_list[w]
                        self.blocks5[idx].active_ch3x3red_out_channel = self.blocks5[
                            idx
                        ].ch3x3red_out_channel_list[w]
                        self.blocks5[idx].active_ch3x3_out_channel = self.blocks5[
                            idx
                        ].ch3x3_out_channel_list[w]
                        self.blocks5[idx].active_ch5x5red_out_channel = self.blocks5[
                            idx
                        ].ch5x5red_out_channel_list[w]
                        self.blocks5[idx].active_ch5x5_out_channel = self.blocks5[
                            idx
                        ].ch5x5_out_channel_list[w]
                        self.blocks5[idx].active_pool_proj_out_channel = self.blocks5[
                            idx
                        ].pool_proj_out_channel_list[w]

    def get_active_subnet_arch(self):
        active_depth_list = []
        active_width_list = []
        active_subwidth_list = []

        active_width_list.append(self.input_stem1[0].active_out_channel)
        active_width_list.append(self.input_stem2[0].active_out_channel)
        for block in self.blocks3:
            active_sub_subwidth_list = []

            if(not isinstance(block, IdentityLayer)):
                active_width_list.append(block.active_out_channel)
                active_sub_subwidth_list.append(block.active_ch1x1_out_channel)
                active_sub_subwidth_list.append(block.active_ch3x3red_out_channel)
                active_sub_subwidth_list.append(block.active_ch3x3_out_channel)
                active_sub_subwidth_list.append(block.active_ch5x5red_out_channel)
                active_sub_subwidth_list.append(block.active_ch5x5_out_channel)
                active_sub_subwidth_list.append(block.active_pool_proj_out_channel)
            active_subwidth_list.append(active_sub_subwidth_list)

        for block in self.blocks4:
            active_sub_subwidth_list=[]

            if(not isinstance(block, IdentityLayer)):
                active_width_list.append(block.active_out_channel)
                active_sub_subwidth_list.append(block.active_ch1x1_out_channel)
                active_sub_subwidth_list.append(block.active_ch3x3red_out_channel)
                active_sub_subwidth_list.append(block.active_ch3x3_out_channel)
                active_sub_subwidth_list.append(block.active_ch5x5red_out_channel)
                active_sub_subwidth_list.append(block.active_ch5x5_out_channel)
                active_sub_subwidth_list.append(block.active_pool_proj_out_channel)
            active_subwidth_list.append(active_sub_subwidth_list)

        for block in self.blocks5:
            active_sub_subwidth_list=[]

            if (not isinstance(block, IdentityLayer)):
                active_width_list.append(block.active_out_channel)
                active_sub_subwidth_list.append(block.active_ch1x1_out_channel)
                active_sub_subwidth_list.append(block.active_ch3x3red_out_channel)
                active_sub_subwidth_list.append(block.active_ch3x3_out_channel)
                active_sub_subwidth_list.append(block.active_ch5x5red_out_channel)
                active_sub_subwidth_list.append(block.active_ch5x5_out_channel)
                active_sub_subwidth_list.append(block.active_pool_proj_out_channel)
            active_subwidth_list.append(active_sub_subwidth_list)

        for depth in self.runtime_depth:
            active_depth_list.append(depth)

        return {
            #"active_kernel_size": active_kernel_size_list,
            "active_depth": active_depth_list,
            #"active_expand_ratio": active_expand_ratio_list,
            "active_width": active_width_list,
            "active_sub_width" : active_subwidth_list,
        }


    def sample_active_subnet(self):

        # sample depth
        #depth_setting = [random.choice([max(self.depth_list), min(self.depth_list)])]   # Was required for Resnet not Inception
        depth_setting = []
        for stage_id in range(len(Inception.BASE_DEPTH_LIST)):
            depth_setting.append(random.choice(self.depth_list))

        # sample width_mult
        width_mult_setting = [
            random.choice(list(range(len(self.input_stem1[0].out_channel_list)))),
            random.choice(list(range(len(self.input_stem2[0].out_channel_list)))),
        ]
        for stage_id, block_idx in enumerate(super().grouped_block_index):
            if (stage_id in [0, 1]):
                stage_first_block = self.blocks3[block_idx[0]]
            if (stage_id in [2, 3, 4, 5, 6]):
                stage_first_block = self.blocks4[block_idx[0]]
            if (stage_id in [7, 8]):
                stage_first_block = self.blocks5[block_idx[0]]
            width_mult_setting.append(
                random.choice(list(range(len(stage_first_block.out_channel_list))))
            )

        arch_config = {"d": depth_setting, "w": width_mult_setting}
        self.set_active_subnet(**arch_config)
        return arch_config

    def get_active_subnet(self, preserve_weight=True):
        input_stem = [self.input_stem1[0].get_active_subnet(3, preserve_weight)]

        input_stem.append(
            self.input_stem2[0].get_active_subnet(
                self.input_stem1[0].active_out_channel, preserve_weight
            )
        )
        input_channel = self.input_stem2[0].active_out_channel

        blocks = []
        for stage_id, block_idx in enumerate(super().grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                if (stage_id in [0, 1]):
                    blocks.append(
                        self.blocks3[idx].get_active_subnet(input_channel, preserve_weight)
                    )
                    input_channel = self.blocks3[idx].active_out_channel
                if (stage_id in [2, 3, 4, 5, 6]):
                    blocks.append(
                        self.blocks4[idx].get_active_subnet(input_channel, preserve_weight)
                    )
                    input_channel = self.blocks4[idx].active_out_channel
                if (stage_id in [7, 8]):
                    blocks.append(
                        self.blocks5[idx].get_active_subnet(input_channel, preserve_weight)
                    )
                    input_channel = self.blocks5[idx].active_out_channel


        classifier = self.classifier.get_active_subnet(input_channel, preserve_weight)
        subnet = Inception(input_stem=input_stem, blocks=blocks, classifier=classifier)

        subnet.set_bn_param(**self.get_bn_param())
        return subnet

    def get_active_net_config(self):
        input_stem_config = [self.input_stem1[0].get_active_subnet_config(3)]

        input_stem_config.append(
            self.input_stem2[0].get_active_subnet_config(
                self.input_stem1[0].active_out_channel
            )
        )
        input_channel = self.input_stem2[0].active_out_channel

        blocks_config = []
        for stage_id, block_idx in enumerate(super().grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                if (stage_id in [0, 1]):
                    blocks_config.append(
                        self.blocks3[idx].get_active_subnet_config(input_channel)
                    )
                    input_channel = self.blocks3[idx].active_out_channel
                if (stage_id in [2, 3, 4, 5, 6]):
                    blocks_config.append(
                        self.blocks4[idx].get_active_subnet_config(input_channel)
                    )
                    input_channel = self.blocks4[idx].active_out_channel
                if (stage_id in [7, 8]):
                    blocks_config.append(
                        self.blocks5[idx].get_active_subnet_config(input_channel)
                    )
                    input_channel = self.blocks5[idx].active_out_channel
        classifier_config = self.classifier.get_active_subnet_config(input_channel)
        return {
            "name": Inception.__name__,
            "bn": self.get_bn_param(),
            "input_stem": input_stem_config[0],
            "input_stem2": input_stem_config[1],
            "blocks3": blocks_config[0:1],
            "blocks4": blocks_config[2:6],
            "blocks5": blocks_config[7:8],
            "dropout": self.dropout_rate,
            "classifier": self.classifier.config,
        }


    def re_organize_middle_weights(self,expand_ratio_stage):  # Write this function


        for block in self.blocks3:
            if not isinstance(block, IdentityLayer):
                block.re_organize_middle_weights()
        for block in self.blocks4:
            if not isinstance(block, IdentityLayer):
                block.re_organize_middle_weights()
        for block in self.blocks5:
            if not isinstance(block, IdentityLayer):
                block.re_organize_middle_weights()


        return



""" Width Related Methods """

'''
    def re_organize_middle_weights(self,expand_ratio_stage):  # Write this function
        return


        for block in self.blocks3:
            block.re_organize_middle_weights()
        for block in self.blocks4:
            block.re_organize_middle_weights()
        for block in self.blocks5:
            block.re_organize_middle_weights()

 1]):
                    blocks_config.append(
                        self.blocks3[idx].get_active_subnet_config(input_channel)
                    )
                    input_channel = self.blocks3[idx].active_out_channel
                if (stage_id in [2, 3, 4, 5, 6]):
                    blocks_config.append(
                        self.blocks4[idx].get_active_subnet_config(input_channel)
                    )
                    input_channel = self.blocks4[idx].active_out_channel
                if (stage_id in [7, 8]):
                    blocks_config.append(
                        self.blocks5[idx].get_active_subnet_config(input_channel)
                    )
                    input_channel = self.blocks5[idx].active_out_channel
        classifier_config = self.classifier.get_active_subnet_config(input_channel)
        return {
            "name": Inception.__name__,
            "bn": self.get_bn_param(),
            "input_stem": input_stem_config[0],
            "input_stem2": input_stem_config[1],
            "blocks3": blocks_config[0:1],
            "blocks4": blocks_config[2:6],
            "blocks5": blocks_config[7:8],
            "dropout": self.dropout_rate,
            "classifier": self.classifier.config,
        }
'''
""" Width Related Methods """

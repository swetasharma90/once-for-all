import torch.nn as nn

from ofa.utils.layers import (
    set_layer_from_config,
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    InceptionBlock,
)
#from ofa.utils.layers import ResNetBottleneckBlock, ResidualBlock
from ofa.utils import make_divisible, MyNetwork, MyGlobalAvgPool2d

__all__ = ["Inception", "GoogleNet"]


class Inception(MyNetwork):

    BASE_DEPTH_LIST = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    STAGE_WIDTH_LIST = [256, 480, 512, 512, 512, 528, 832, 832, 1024]
    # 1x1 3x3reduce 3x3 5x5reduce 5x5 poolproj
    STAGE_WIDTH_SUB_LIST = [[64, 96, 128, 16, 32, 32],
                            [128, 128, 192, 32, 96, 64],
                            [192, 96, 208, 16, 48, 64],
                            [160, 112, 224, 24, 64, 64],
                            [128, 128, 256, 24, 64, 64],
                            [112, 144, 288, 32, 64, 64],
                            [256, 160, 320, 32, 128, 128],
                            [256, 160, 320, 32, 128, 128],
                            [384, 192, 384, 48, 128, 128]]
    for width,sub_width in zip(STAGE_WIDTH_LIST,STAGE_WIDTH_SUB_LIST):
        list_sum = []
        for j in [0, 2, 4, 5]:
            list_sum.append(sub_width[j])
        assert width==sum(list_sum), 'Stage width and Subwidth sum unequal'

    def __init__(self, input_stem, blocks, _dropout = True, dropout_rate=0.4, classifier = None):
        super(Inception, self).__init__()

        self.input_stem1 = nn.ModuleList(input_stem[0])
        self.max_pooling1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.input_stem2 = nn.ModuleList(input_stem[1])
        self.max_pooling2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        #for blocks_ in blocks[0]:
        self.blocks3 = nn.ModuleList(blocks[0])
        #self.blocks3.append(blocks[1])
        for blocks_ in blocks[1]:
            self.blocks3.append(blocks_)
        self.max_pooling3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.blocks4 = nn.ModuleList(blocks[2])
        for i in range(3,7):
            for blocks_ in blocks[i]:
                self.blocks4.append(blocks_)

        self.max_pooling4 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.blocks5 = nn.ModuleList(blocks[7])
        for blocks_ in blocks[8]:
            self.blocks5.append(blocks_)
        self.avg_pool = MyGlobalAvgPool2d(keep_dim=False)
        #nn.AvgPool2d(kernel_size = 7, stride = 1, padding=1, ceil_mode=False)
        if(_dropout):
            self.dropout_rate = dropout_rate
            self.dropout = nn.Dropout(self.dropout_rate, inplace = True)
        self.classifier = classifier

    def forward(self, x):
        for layer in self.input_stem1:
            x = layer(x)
        x = self.max_pooling1(x)
        for layer in self.input_stem2:
            x = layer(x)
        x = self.max_pooling2(x)
        for block in self.blocks3:
            x = block(x)
        x = self.max_pooling3(x)
        for block in self.blocks4:
            x = block(x)
        x = self.max_pooling4(x)
        for block in self.blocks5:
            x = block(x)
        x = self.avg_pool(x)
        if (self.dropout_rate>0):
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
        for block in self.blocks3:
            _str += block.module_str + "\n"
        _str += "max_pooling(ks=3, stride=2)\n"
        for block in self.blocks4:
            _str += block.module_str + "\n"
        _str += "max_pooling(ks=3, stride=2)\n"
        for block in self.blocks5:
            _str += block.module_str + "\n"
        #_str += "max_pooling(ks=3, stride=2)\n"
        _str += self.avg_pool.__repr__() + "\n"   # Check if __repr__() should be replaced by module_str
        _str += self.dropout.__repr__() + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": Inception.__name__,
            "bn": self.get_bn_param(),  # Check what is this doing
            "input_stem1": [layer.config for layer in self.input_stem1],
            "input_stem2": [layer.config for layer in self.input_stem2],
            "blocks3": [block.config for block in self.blocks3],
            "blocks4": [block.config for block in self.blocks4],
            "blocks5": [block.config for block in self.blocks5],
            "dropout": self.dropout_rate,
            "classifier": self.classifier.config,
        }

    @staticmethod # Write this function
    def build_from_config(config): # Write this function
        classifier = set_layer_from_config(config["classifier"])

        input_stem1 = []
        for layer_config in config["input_stem1"]:
            input_stem1.append(set_layer_from_config(layer_config)) # Implement this method for Inception Block
        input_stem2 = []
        for layer_config in config["input_stem2"]:
            input_stem2.append(set_layer_from_config(layer_config))
        blocks3 = []
        for block_config in config["blocks3"]:
            blocks3.append(set_layer_from_config(block_config))
        blocks4 = []
        for block_config in config["blocks4"]:
            blocks4.append(set_layer_from_config(block_config))
        blocks5 = []
        for block_config in config["blocks5"]:
            blocks5.append(set_layer_from_config(block_config))
        dropout_rate = config["dropout"]
        net = Inception([input_stem1, input_stem2], [blocks3, blocks4, blocks5], _dropout = True, dropout_rate = dropout_rate, classifier = classifier)
        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)

        return net

    '''
    def zero_last_gamma(self): # Check what is this function doing and edit accordingly
        for m in self.modules():
            if isinstance(m, ResNetBottleneckBlock) and isinstance(
                m.downsample, IdentityLayer
            ):
                m.conv3.bn.weight.data.zero_()
    '''
    @property
    def grouped_block_index(self):   # Check the role of Identity and modify this function accordigly
        info_list = []              # Identity layer acts as a marker between two stages and maintains indexes to
        block_index_list = []       # individual blocks within each stage
                                    # Used for only DYNAMIC INCEPTION Networks
        for i, block in enumerate(self.blocks3):
            #print(block)
            if (
                isinstance(block, IdentityLayer)
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
                isinstance(block, IdentityLayer)
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
                isinstance(block, IdentityLayer)
                and len(block_index_list) > 0
            ):
                info_list.append(block_index_list)
                block_index_list = []
            else:
                block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    def load_state_dict(self, state_dict, **kwargs):
        super(Inception, self).load_state_dict(state_dict)


class GoogleNet(Inception):
    def __init__(
        self,
        n_classes=1000,
        width_mult=1.0,
        bn_param=(0.1, 1e-5),
        dropout_rate=0.4,  # 40% for GoogleNet
        expand_ratio=None,
        depth_param=None,
    ):

        #expand_ratio = 0.25 if expand_ratio is None else expand_ratio

        input_channel = make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
        stage_width_list = Inception.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = make_divisible(
                width * width_mult, MyNetwork.CHANNEL_DIVISIBLE
            )
        stage_width_sub_list = Inception.STAGE_WIDTH_SUB_LIST.copy()
        for i, width in enumerate(stage_width_sub_list):
            stage_width_sub_list[i] = [make_divisible(
                _width * width_mult, MyNetwork.CHANNEL_DIVISIBLE) for _width in stage_width_sub_list[i]
            ]


        depth_list = Inception.BASE_DEPTH_LIST.copy() #[3, 4, 6, 3]
        if depth_param is not None:
            for i, depth in enumerate(Inception.BASE_DEPTH_LIST):
                depth_list[i] = depth + depth_param

        stride_list = [1, 1, 1, 1, 1, 1, 1, 1, 1]

        # build input stem
        input_stem1 = [
            ConvLayer(
                3,
                input_channel,
                kernel_size=7,
                stride=2,
                use_bn=True,
                act_func="relu",
                ops_order="weight_bn_act", # For now use Batch Normalization but test without batch norm as well.
            )
        ]

        input_stem2 = [
            ConvLayer(
                input_channel,
                input_channel*3,  # Hard-coded for now
                kernel_size=3,
                stride=1,
                use_bn=True,
                act_func="relu",
                ops_order="weight_bn_act",  # For now use Batch Normalization but test without batch norm as well.
            )
        ]

        # blocks
        blocks = []
         # Write this code
        for d, width, sub_width, s in zip(depth_list, stage_width_list, stage_width_sub_list, stride_list):
            for i in range(d):
               # stride = s if i == 0 else 1
                inception_block = InceptionBlock(
                    input_channel,
                    width,
                    sub_width[0],
                    sub_width[1],
                    sub_width[2],
                    sub_width[3],
                    sub_width[4],
                    sub_width[5],
                    stride=s,
                )
                blocks.append(inception_block)
                input_channel = width
            blocks.append(IdentityLayer())  # Append identity layer to signify end of a stage

        # classifier
        classifier = LinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)

        super(GoogleNet, self).__init__([input_stem1,input_stem2], blocks, dropout_rate = self.dropout_rate,
                                        classifier = classifier)

        # set bn param
        self.set_bn_param(*bn_param)



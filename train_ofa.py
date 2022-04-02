import os, argparse, random
from datetime import datetime
import numpy as np

import torch

from ofa.imagenet_classification.run_manager.run_config import Cifar100RunConfig, TinyImagenetRunConfig
from ofa.imagenet_classification.run_manager.run_manager import RunManager
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3, OFAResNets, OFAInception
from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import validate, load_models
from ofa.utils import MyRandomResizedCrop

from utils import loadConfigs

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='depth', choices=['kernel', 'depth', 'expand', 'teacher','width'])
parser.add_argument('--dataset', type=str, default="cifar100")
parser.add_argument('--model', type=str, default="OFAMobileNetV3")
parser.add_argument('--read_path', type=str)
parser.add_argument('--n_worker', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--image_size', type=str)
parser.add_argument('--ks_list', type=str, default="7")
parser.add_argument('--expand_list', type=str, default="6")
parser.add_argument('--depth_list', type=str, default="4")
parser.add_argument('--width_mult_list', type=str, default="1.0")

args = parser.parse_args()

if args.task == 'teacher':

    args.path = os.path.join("./exp", str(datetime.now()).replace(":", "."), "teacher")
    os.makedirs(args.path, exist_ok=True)
    args.image_size = [int(img_size) for img_size in args.image_size.split(',')]
    args.teacher_model = None
    args.kd_ratio = 0
    args.dynamic_batch_size = 1
    args.init_lr = 6e-2

else:

    args.teacher_path = os.path.join(args.read_path, "teacher/checkpoint/model_best.pth.tar")
    teacher_run_config = loadConfigs(os.path.join(args.read_path, "teacher/run.config"), ["image_size", "dataset"])
    args.image_size , args.dataset = teacher_run_config["image_size"], teacher_run_config["dataset"]
    args.model = loadConfigs(os.path.join(args.read_path, "teacher/net.config"), ["name"])["name"]
    args.kd_ratio = 1.0
    args.kd_type = "ce"

    if args.task == 'kernel':
        args.path = os.path.join(args.read_path, "elastic kernel")
        args.dynamic_batch_size = 1
        args.init_lr = 3e-2
        #args.ks_list = '3,5,7'
        #args.expand_list = '6'
        #args.depth_list = '4'
        #args.width_mult_list = '1.0'
    elif args.task == 'depth':
        args.path = os.path.join(args.read_path, "elastic depth")
        args.dynamic_batch_size = 2
        args.init_lr = 7.5e-3
        #args.ks_list = '3,5,7'
        #args.expand_list = '6'
        #args.depth_list = '2,3,4'
        #args.width_mult_list = '1.0'
    elif args.task == 'expand':
        args.path = os.path.join(args.read_path, "elastic expand")
        args.dynamic_batch_size = 4
        args.init_lr = 7.5e-3
        #args.ks_list = '3,5,7'
        #args.expand_list = '3,4,6'
        #args.depth_list = '2,3,4'
        #args.width_mult_list = '1.0'
    elif args.task == 'width':
        args.path = os.path.join(args.read_path, "elastic width")
        args.dynamic_batch_size = 4
        args.init_lr = 7.5e-3
        #args.ks_list = '3,5,7'
        #args.expand_list = '3,4,6'
        #args.depth_list = '2,3,4'
        #args.width_mult_list = '1.0'
    else:
        raise NotImplementedError

# ofa_resnet50_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0

args.manual_seed = 0

args.valid_size = 0.1
args.validation_frequency = 1
args.print_frequency = 10

args.warmup_epochs = 5
args.warmup_lr = -1
args.opt_type = 'sgd'
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = 'bn#bias'
args.fp16_allreduce = False
args.model_init = 'he_fout'

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.2


#args.dy_conv_scaling_mode = 1
#args.continuous_size = False
#if args.dy_conv_scaling_mode == -1:
#        args.dy_conv_scaling_mode = None
    
DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1
MyRandomResizedCrop.CONTINUOUS = False


if __name__ == '__main__':

    print(args.path)
    os.makedirs(args.path, exist_ok=True)

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    if args.warmup_lr < 0: args.warmup_lr = args.init_lr
    args.train_batch_size = args.batch_size
    args.test_batch_size = args.batch_size * 4
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    
    if args.dataset == "cifar100":
        run_config = Cifar100RunConfig(**args.__dict__)
    elif args.dataset == "tiny_imagenet":
        run_config = TinyImagenetRunConfig(**args.__dict__)
    else:
        raise NotImplementedError
    
    # build net from args
    args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list = [float(e) for e in args.expand_list.split(',')]
    args.depth_list = [int(d) for d in args.depth_list.split(',')]
    args.width_mult_list = [float(width_mult) for width_mult in args.width_mult_list.split(',')]
    args.width_mult_list = args.width_mult_list[0] if len(args.width_mult_list) == 1 else args.width_mult_list
    #print(args.width_mult_list)
    if args.model == OFAMobileNetV3.__name__:
        net = OFAMobileNetV3(
            n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=args.dropout, width_mult=args.width_mult_list,
            ks_list=args.ks_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list
        )
    elif args.model == OFAResNets.__name__:
        net = OFAResNets(
            n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
            ks_list=args.ks_list, dropout_rate=args.dropout, width_mult_list=args.width_mult_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list
        )
    elif args.model == OFAInception.__name__:
        net = OFAInception(
            n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=args.dropout, width_mult_list=args.width_mult_list, depth_list=args.depth_list
        )
    else:
        raise NotImplementedError

    run_manager = RunManager(args.path, net, run_config, is_root=True,device=torch.device("cuda"))
    run_manager.save_config()

    if args.task == "teacher":

        run_manager.train(args, warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr)

    else:

        validate_func_dict = {
            'image_size_list': {args.image_size} if isinstance(args.image_size, int) else {min(args.image_size), max(args.image_size)},
            'ks_list': sorted({min(args.ks_list), max(args.ks_list)}),
            'expand_ratio_list': sorted({min(args.expand_list), max(args.expand_list)}),
            'depth_list': sorted({min(net.depth_list), max(net.depth_list)}),
            'width_mult_list': sorted({min(net.width_mult_list), max(net.width_mult_list)})
        }

        if args.model == OFAMobileNetV3.__name__:
            args.teacher_model = OFAMobileNetV3(
                n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=0, base_stage_width=args.base_stage_width, width_mult=1.0,
                ks_list=[7], expand_ratio_list=[6], depth_list=[4]
            )
        elif args.model == OFAResNets.__name__:
            args.teacher_model = OFAResNets(
                n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
                ks_list=[7], dropout_rate=0., width_mult_list=[1.0], expand_ratio_list=[0.35], depth_list=[2]
            )
        elif args.model == OFAInception.__name__:
            args.teacher_model = OFAInception(
                n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout, width_mult_list=args.width_mult_list, depth_list=args.depth_list
            )
        else:
            raise NotImplementedError

        args.teacher_model.cuda()
        load_models(run_manager, args.teacher_model, model_path=args.teacher_path)

        from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import train

        # KERNEL SIZE
        if args.task == 'kernel':
            validate_func_dict['ks_list'] = sorted(args.ks_list)
            load_models(run_manager, net, model_path=args.teacher_path)
            train(run_manager, args,
                lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict))
        
        



        
        # LAYER PER UNIT
        elif args.task == 'depth':
            if args.model == OFAInception.__name__:
                from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import train_elastic_depth
                args.ofa_checkpoint_path = os.path.join(args.read_path, "teacher/checkpoint/model_best.pth.tar")
                train_elastic_depth(train, run_manager, args, validate_func_dict)
            else:
                from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import train_elastic_depth

                args.ofa_checkpoint_path = os.path.join(args.read_path, "elastic kernel/checkpoint/model_best.pth.tar")
                train_elastic_depth(train, run_manager, args, validate_func_dict)
        
        
        
        
        

        # EXPAND INTRA BLOCK
        elif args.task == 'expand':
            from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import train_elastic_expand
            args.ofa_checkpoint_path = os.path.join(args.read_path, "elastic depth/checkpoint/model_best.pth.tar")
            train_elastic_expand(train, run_manager, args, validate_func_dict)

        elif args.task == 'width':
            if args.model == OFAInception.__name__:
                validate_func_dict = {
                    'image_size_list': {args.image_size} if isinstance(args.image_size, int) else {min(args.image_size),
                                                                                                   max(args.image_size)},
                    'ks_list': sorted({min(args.ks_list), max(args.ks_list)}),
                    'expand_ratio_list': sorted({min(args.expand_list), max(args.expand_list)}),
                    'depth_list': sorted(net.depth_list),
                    'width_mult_list': sorted(net.width_mult_list)
                }
                from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import train_elastic_width_mult

                args.ofa_checkpoint_path = os.path.join(args.read_path, "elastic depth/checkpoint/model_best.pth.tar")
                train_elastic_width_mult(train, run_manager, args, validate_func_dict)
            else:
                from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import train_elastic_width_mult

                args.ofa_checkpoint_path = os.path.join(args.read_path, "elastic expand/checkpoint/model_best.pth.tar")
                train_elastic_width_mult(train, run_manager, args, validate_func_dict)

        
        
        
        
        
        
        else:
            raise NotImplementedError

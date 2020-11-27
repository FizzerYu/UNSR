import torch
import argparse
from importlib import import_module
import sys
import time 
import numpy as np 

def running_time(net,args):
    net = net.cuda()
    x = torch.rand(1, args.n_colors, args.patch_size, args.patch_size)
    x = x.cuda()
    y = net(x)
    total_time = []
    print('================> start')
    for i in range(1000):
        start = time.time()
        y = net(x)
        end = time.time()
        total_time.append(end-start)
    print('time usage average {:3f}, min {}, max {}'.format(np.mean(total_time), np.min(total_time), np.max(total_time)))

def model_summary(net):
    print('===> model summary')
    try:
        from torchsummary import summary
        summary(net, (3, int(args.patch_size/args.scale[0]), int(args.patch_size/args.scale[0])), depth=8, verbose=1, 
                            col_names=["input_size", "output_size","kernel_size", "num_params","mult_adds"])
    except ImportError as error:
        # Output expected ImportErrors.
        print(error.__class__.__name__ + ": " + error.message)
        print('Need install torch-summary from "pip install torch-summary" ')
        print(net)
    except Exception:
        print('Need install torch-summary from "pip install torch-summary" not "torchsummary"')
        print(net)

parser = argparse.ArgumentParser(description = 'EVALUATION MODEL TIME')
parser.add_argument('--scale',type=str,default='4')
parser.add_argument('--n_colors',type=int,default=3)
parser.add_argument('--model',type=str,default='.')
parser.add_argument('--model_path',type=str,default='/home/chenjunming/fizzer/PFFN/code/model/')
parser.add_argument('--model_choose',type=str,default='RCAN')
parser.add_argument('--patch_size', type=int, default=48, help='input patch size')

if __name__ == '__main__':
    args = parser.parse_args()

    args.scale = list(map(lambda x: int(x), args.scale.split('+')))    
    sys.path.append(args.model_path)
    module = import_module(args.model)
    model = module.make_model(args)  # 加载模型
    running_time(model, args)



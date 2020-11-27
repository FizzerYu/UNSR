import os

# from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        for entry in os.scandir(self.dir_hr):    # 目录迭代方法
            filename = os.path.splitext(entry.name)[0]
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    '{}{}'.format(filename, self.ext)  # 注意文件名
                ))

        list_hr.sort()
        for l in list_lr:
            l.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):   # args.dir_data  
        #######################################################
        # 这里按照需求修改
        # 现在
        # |-- /titan_data2/lichangyu/sr_dataset
        #     |-- HR
        #         |-- B100
        #             |-- x2
        #             |-- x3
        #             |-- x4
        #     |-- LR/LRBI
        #         |-- B100
        #             |-- x2
        #             |-- x3
        #             |-- x4
        #######################################################
        # self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.dir_hr = os.path.join(dir_data, 'HR', self.args.data_test, 'x{}'.format(self.scale[0]))
        self.dir_lr = os.path.join(dir_data, 'LR/LRBI', self.args.data_test, 'x{}'.format(self.scale[0]))
        self.ext = '.png'

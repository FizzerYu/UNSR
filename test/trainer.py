import os
import math
from decimal import Decimal
import time

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):  

        self.args = args
        self.scale = args.scale

        self.ckp = ckp

        self.loader_test = loader.loader_test
        self.model = my_model

        self.error_last = 1e8


    def test(self):

        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:

                        lr=lr.to('cuda',non_blocking=True)
                        hr=hr.to('cuda',non_blocking=True)
                    else:

                        lr=lr.to('cuda',non_blocking=True)

                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]

                    if sr.shape != hr.shape: 
                        print('\n', filename, lr.shape, sr.shape, hr.shape)
                        # raise 

                    save_list.extend([lr, hr])

                    if self.args.save_results:

                        self.ckp.save_results_nopostfix(filename, save_list, scale)   # 预测的图片存起来

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device,non_blocking=True)
           
        return [_prepare(_l) for _l in l]


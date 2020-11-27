import torch

import utility
import data
import model
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)  # here we use link files to linke the train and test
    if not args.test_only:
        utility.illegal()  # illegal operation in test folder
    else:
        loss = None
    t = Trainer(args, loader, model, loss, checkpoint)

    t.test()
    
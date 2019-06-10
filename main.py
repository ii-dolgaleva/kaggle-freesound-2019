import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
import copy

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from lr_finder import LRFinder
import matplotlib.pyplot as plt


import data
import models
import train
import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/data/kaggle-freesound-2019')
    parser.add_argument('--outpath', default='/data/runs/')
    parser.add_argument('--epochs', default=20)
    parser.add_argument('--batch_size', default=32)
    return parser.parse_args()


def main(args):
    np.random.seed(432)
    torch.random.manual_seed(432)
    try:
        os.makedirs(args.outpath)
    except OSError:
        pass
    experiment_path = utils.get_new_model_path(args.outpath)
    print(experiment_path)

    train_writer = SummaryWriter(os.path.join(experiment_path, 'train_logs'))
    val_writer = SummaryWriter(os.path.join(experiment_path, 'val_logs'))
    trainer = train.Trainer(train_writer, val_writer)

    # todo: add config
    train_transform = data.build_preprocessing()
    eval_transform = data.build_preprocessing()

    trainds, evalds = data.build_dataset(args.datadir, None)
    trainds.transform = train_transform
    evalds.transform = eval_transform

    
    model = models.resnet34()
    opt = torch.optim.Adam(model.parameters(), lr=1e-8)

    trainloader = DataLoader(trainds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    evalloader = DataLoader(evalds, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    
    #find lr fast ai
    criterion = torch.nn.BCEWithLogitsLoss()
    lr_finder = LRFinder(model, opt, criterion, device="cuda")
#     lr_finder.range_test(trainloader, val_loader=evalloader, end_lr=1, num_iter=10, step_mode="exp")
    lr_finder.range_test(trainloader, end_lr=100, num_iter=100, step_mode="exp")
    
    #plot graph fast ai
    skip_start=6
    skip_end=3
    lrs = lr_finder.history["lr"]
    losses = lr_finder.history["loss"]
    grad_norm = lr_finder.history["grad_norm"]
    
#     ind = grad_norm.index(min(grad_norm))
#     opt_lr = lrs[ind]
#     print('LR with min grad_norm =', opt_lr)
    
    lrs = lrs[skip_start:-skip_end]
    losses = losses[skip_start:-skip_end]
    
    
    fig = plt.figure(figsize=(12, 9))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    train_writer.add_figure('loss_vs_lr', fig)
    
    lr_finder.reset()

    fixed_lr = 1e-3
#     fixed_lr = 3e-4
    opt = torch.optim.Adam(model.parameters(), lr=fixed_lr)
    
#     #new
#     lr = 1e-3
#     eta_min = 1e-5
#     t_max = 10
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)
#     #new
    
#     one cycle for 5 ehoches
    scheduler = CosineAnnealingLR(opt, 519*4, eta_min=1e-4)
#     scheduler = CosineAnnealingLR(opt, 519, eta_min=1e-5)
#     scheduler = StepLR(opt, step_size=3, gamma=0.1)
      
    state_list = []
    for epoch in range(args.epochs):
#         t = epoch / args.epochs
#         lr = np.exp((1 - t) * np.log(lr_begin) + t * np.log(lr_end))
        # выставляем lr для всех параметров
        trainer.train_epoch(model, opt, trainloader, fixed_lr, scheduler)
#         trainer.train_epoch(model, opt, trainloader, 3e-4, scheduler)
#         trainer.train_epoch(model, opt, trainloader, 9.0451e-4, scheduler)
        metrics = trainer.eval_epoch(model, evalloader)

        state = dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=opt.state_dict(),
            loss=metrics['loss'],
            lwlrap=metrics['lwlrap'],
            global_step=trainer.global_step,
        )
        state_copy = copy.deepcopy(state)
        state_list.append(state_copy)
        export_path = os.path.join(experiment_path, 'last.pth')
        torch.save(state, export_path)

    # save the best path
    best_export_path = os.path.join(experiment_path, 'best.pth')
    
    max_lwlrap = 0
    max_lwlrap_ind = 0
    for i in range(args.epochs):
        if state_list[i]['lwlrap'] > max_lwlrap:
            max_lwlrap = state_list[i]['lwlrap']
            max_lwlrap_ind = i
    
    best_state = state_list[max_lwlrap_ind]
    torch.save(best_state, best_export_path)
        
if __name__ == "__main__":
    args = _parse_args()
    main(args)

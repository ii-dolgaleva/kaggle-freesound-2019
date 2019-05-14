import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader

import data
import models
import train
import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/data/kaggle-freesound-2019')
    parser.add_argument('--outpath', default='/data/runs/')
    parser.add_argument('--epochs', default=10)
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
    opt = torch.optim.Adam(model.parameters())

    trainloader = DataLoader(trainds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    evalloader = DataLoader(evalds, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    for epoch in range(args.epochs):
        trainer.train_epoch(model, opt, trainloader, 3e-4)
        metrics = trainer.eval_epoch(model, evalloader)

        state = dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=opt.state_dict(),
            loss=metrics['loss'],
            lwlrap=metrics['lwlrap'],
            global_step=trainer.global_step,
        )
        export_path = os.path.join(experiment_path, 'last.pth')
        torch.save(state, export_path)


if __name__ == "__main__":
    args = _parse_args()
    main(args)

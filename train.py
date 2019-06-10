from collections import defaultdict
from sklearn.metrics import label_ranking_average_precision_score
import torch
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import utils

plt.switch_backend('agg')


def apply_wd(model, gamma):
    for name, tensor in model.named_parameters():
        if 'bias' in name:
            continue
        tensor.data.add_(-gamma * tensor.data)


def grad_norm(model):
    grad = 0.0
    count = 0
    for name, tensor in model.named_parameters():
        if tensor.grad is not None:
            grad += torch.sqrt(torch.sum((tensor.grad.data) ** 2))
            count += 1
    return grad.cpu().numpy() / count


class Trainer:
    global_step = 0

    def __init__(self, train_writer=None, eval_writer=None, compute_grads=True, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self.device = device
        self.train_writer = train_writer
        self.eval_writer = eval_writer
        self.compute_grads = compute_grads

    def train_epoch(self, model, optimizer, dataloader, lr, scheduler, log_prefix=""):
        device = self.device

        model = model.to(device)
        model.train()
#         scheduler.step()

        logs = {}
        logs['lrap'] = []
        logs['loss'] = []
        logs['lr'] = []
        
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = scheduler.get_lr()[0]

#             if param_group['lr'] < 1.2653e-4:
#                 param_group['lr'] = 1.2653e-4
#             param_group['lr'] = lr

        print('scheduler.get_lr',scheduler.get_lr()[0])
        print('optimizer.param_groups', optimizer.param_groups[0]['lr'])
        
        for batch in tqdm(dataloader):
            x = batch['logmel'].to(device)
            y = batch['labels'].to(device)
            
            scheduler.step()
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler.get_lr()[0]

            optimizer.zero_grad()
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out, y)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(out).cpu().data.numpy()
            lrap = label_ranking_average_precision_score(batch['labels'], probs)
            log_entry = dict(
                lrap=lrap,
                loss=loss.item(),
#                 lr=lr,
                lr=scheduler.get_lr()[0],
            )
            
            logs['lrap'].append(lrap)
            logs['loss'].append(loss.item())
            logs['lr'].append(scheduler.get_lr()[0])
            
            if self.compute_grads:
                log_entry['grad_norm'] = grad_norm(model)

            for name, value in log_entry.items():
                if log_prefix != '':
                    name = log_prefix + '/' + name
                self.train_writer.add_scalar(name, value, global_step=self.global_step)
            
            self.global_step += 1
            
        fig = plt.figure(figsize=(12, 9))
        plt.plot(logs['lr'], logs['loss'])
        plt.xscale('log')
        plt.grid()
        self.train_writer.add_figure('loss_vs_lr_for_batch', fig, global_step=self.global_step)

    def eval_epoch(self, model, dataloader, log_prefix=""):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        model = model.to(device)
        model.eval()
        metrics = defaultdict(list)
        lwlrap = utils.lwlrap_accumulator()
        for batch in tqdm(dataloader):
            with torch.no_grad():
                x = batch['logmel'].to(device)
                y = batch['labels'].to(device)
                out = model(x)
                loss = F.binary_cross_entropy_with_logits(out, y)
                probs = torch.sigmoid(out).cpu().data.numpy()
                lrap = label_ranking_average_precision_score(batch['labels'], probs)
                lwlrap.accumulate_samples(batch['labels'], probs)

                metrics['loss'].append(loss.item())
                metrics['lrap'].append(lrap)

        metrics = {key: np.mean(values) for key, values in metrics.items()}
        metrics['lwlrap'] = lwlrap.overall_lwlrap()
        for name, value in metrics.items():
            if log_prefix != '':
                name = log_prefix + '/' + name
            self.eval_writer.add_scalar(name, value, global_step=self.global_step)

        fig = plt.figure(figsize=(12, 9))
        z = lwlrap.per_class_lwlrap() * lwlrap.per_class_weight()
        plt.bar(np.arange(len(z)), z)
        plt.hlines(np.mean(z), 0, len(z), linestyles='dashed')
        plt.ylim([0, 0.013])
        plt.xlim([-1, 80])
        plt.grid()
        self.eval_writer.add_figure('per_class_weighted_lwlrap', fig, global_step=self.global_step)

        fig = plt.figure(figsize=(12, 9))
        z = lwlrap.per_class_lwlrap()
        plt.bar(np.arange(len(z)), z)
        plt.hlines(np.mean(z), 0, len(z), linestyles='dashed')
        plt.xlim([-1, 80])
        plt.grid()
        self.eval_writer.add_figure('per_class_lwlrap', fig, global_step=self.global_step)

        return metrics

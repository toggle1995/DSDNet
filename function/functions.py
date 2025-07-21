import os
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import torch.nn as nn

def save_checkpoint(paras, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, log_dir, best_flag=False):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': paras}

    # save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    if best_flag:
        save_path = os.path.join(log_dir, 'best_parameter.pth')
    else:
        save_path = os.path.join(log_dir, 'saved_parameter.pth')
    logger(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger(f"{save_path} saved !!!")


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    log_softmax_targets = F.log_softmax(targets, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    softmax_outputs = F.softmax(outputs, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def InfoEnergy(outputs):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_outputs = F.softmax(outputs, dim=1)
    return -(log_softmax_outputs * softmax_outputs).sum(dim=1).mean()


class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        reduction='batchmean',
    ):
        super(KLDivergence, self).__init__()
        self.tau = tau

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        return loss


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        log_string(f"RANK and WORLD_SIZE in environment: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    setup_for_distributed(is_main_process())

    if args.output_dir:
        mkdir(args.output_dir)
    if args.model_id:
        mkdir(os.path.join('./models/', args.model_id))
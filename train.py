import torch
import numpy as np 
import cv2
import os
from function import functions
from torch.utils.data import DataLoader
import dataset_plain
import model
import argparse
from PIL import Image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch import nn
from torchvision import models
import torch.nn.functional as F
import shutil
import pdb
import time
import sys
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import accuracy, AverageMeter
import csv
import utils
from sklearn.metrics import cohen_kappa_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 32]')
parser.add_argument('--root_dir', default='log', help='Root dir [default: log]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--data_root', default='./', help='Where to load the data [default: ./]')
parser.add_argument('--base_lr', type=float, default=2e-5, help='Initial learning rate [default: 0.01]')
parser.add_argument('--base_warmup_lr', type=float, default=1e-6, help='Warm-up learning rate [default: 0.01]')
parser.add_argument('--base_min_lr', type=float, default=1e-5, help='Min learning rate [default: 0.01]')
parser.add_argument("--wd", type=float, default=5e-4, help='weight_decay utlized in the training')
parser.add_argument('--lr_decay', type=int, default=40)
parser.add_argument('--total_epoch', type=int, default=30)
parser.add_argument('--optim', default='momentum', help='momentum, adam or rmsp [defualt: momentum]')
parser.add_argument('--backbone_name', default='res50', help='the name of backbone network')
parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
parser.add_argument("--train_index", type=int, default=0, help='which split is utlized in the training')

args = parser.parse_args()
batch_size = args.batch_size
base_learning_rate = args.base_lr
log_dir = args.log_dir
optim = args.optim
gpu_index = args.gpu
lr_decay = args.lr_decay
total_epoch = args.total_epoch
wd = args.wd
backbone_name = args.backbone_name

utils.init_distributed_mode(args)
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
start_time = time.strftime('%Y-%m-%d.%H:%M:%S',time.localtime(time.time()))

np.random.seed(0)
cudnn.benchmark = True
torch.manual_seed(0)
cudnn.enabled=True
torch.cuda.manual_seed(0)

log_dir = os.path.join(args.root_dir, log_dir)
name_file = sys.argv[0]
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
os.system('cp %s %s' % (name_file, log_dir))
os.system('cp %s %s' % ('model.py', log_dir))
os.system('cp %s %s' % ('dataset.py', log_dir))
LOG_FOUT = open(os.path.join(log_dir, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')
LOG_FOUT.write(start_time+'\n')
print(str(args))
print('The time now is:' + start_time)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)

    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.normal(m.weight.data, mean=0, std=1)
        m.bias.data.zero_()

def weight_init1(m):
    if isinstance(m, nn.Linear):
        nn.init.normal(m.weight.data, mean=0, std=0.01)
        m.bias.data.zero_()


def calculate_metric(predictions, ground_truths, num_classes=1000):
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    for i in range(len(ground_truths)):
        total[ground_truths[i]] += 1
        if predictions[i] == ground_truths[i]:
            correct[ground_truths[i]] += 1
    per_class_accuracy = correct / total
    average_accuracy = np.nanmean(per_class_accuracy)
    kappa = cohen_kappa_score(ground_truths, predictions)
    return average_accuracy, kappa

def get_model(model_name):
    try:
        model_class = getattr(model, model_name)
        return model_class()
    except AttributeError:
        raise ValueError(f"Model {model_name} not found in the module.")

def main():
    train_file_path = os.path.join(args.data_root, 'train_split_{}.txt'.format(args.train_index))
    val_file_path = os.path.join(args.data_root, 'test_split_{}.txt'.format(args.train_index))
    train_data = dataset_plain.FGB(train_file_path, is_training=True)
    test_data = dataset_plain.FGB(val_file_path, is_training=False)

    log_string(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(test_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=256, sampler=test_sampler, num_workers=4, drop_last=False)


    train_num_steps = len(train_loader)
    
    learning_rate = args.base_lr * batch_size / 32
    warmup_lr = args.base_warmup_lr * batch_size / 32
    min_lr = args.base_min_lr * batch_size / 32

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model_name = 'dsd_{}'.format(backbone_name)
    net = get_model(model_name)

    LOG_FOUT = open(os.path.join(log_dir, 'model_record.txt'), 'w')
    LOG_FOUT.write(str(net))
    LOG_FOUT.flush()

    net.cuda().train()

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], find_unused_parameters=True)

    loss_func = nn.CrossEntropyLoss().cuda()
    loss_ce = functions.CrossEntropy
    loss_ie = functions.InfoEnergy
    

    backbone_params = list(net.module.layer0.parameters()) + \
                  list(net.module.layer1.parameters()) + \
                  list(net.module.layer2.parameters()) + \
                  list(net.module.layer3.parameters()) + \
                  list(net.module.layer4.parameters())

    fc_params = list(net.module.fc.parameters()) + \
                list(net.module.fc0.parameters()) + \
                list(net.module.fc1.parameters()) + \
                list(net.module.fc2.parameters()) + \
                list(net.module.fc3.parameters())

    backbone_param_names = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
    fc_param_names = ['fc', 'fc0', 'fc1', 'fc2', 'fc3']
    other_params = [param for name, param in net.module.named_parameters() if not any(backbone_name in name for backbone_name in (backbone_param_names + fc_param_names))]


    if optim == 'momentum':
        optimizer = torch.optim.SGD([
            {'params': backbone_params, 'lr': learning_rate},
            {'params': other_params, 'lr': learning_rate * 5}
        ], momentum=0.9, weight_decay=wd, nesterov=True)
    elif optim == 'adam':
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': learning_rate},
            {'params': other_params, 'lr': learning_rate * 5}
        ], weight_decay=wd)
    elif optim == 'rmsp':
        optimizer = torch.optim.RMSprop([
            {'params': backbone_params, 'lr': learning_rate},
            {'params': fc_params, 'lr': learning_rate * 10},
            {'params': other_params, 'lr': learning_rate}
        ], weight_decay=wd)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), learning_rate, weight_decay=wd)
    
    TRAIN_WARMUP_EPOCHS = 5
    TRAIN_LR_SCHEDULER_WARMUP_PREFIX = True
    warmup_steps = int(TRAIN_WARMUP_EPOCHS * train_num_steps)
    total_num_steps = int(total_epoch * train_num_steps)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=((total_num_steps - warmup_steps) // 8 ) if TRAIN_LR_SCHEDULER_WARMUP_PREFIX else train_num_steps // 8,
        # t_mul=1.,
        lr_min=min_lr,
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=8,
        t_in_epochs=False,
        warmup_prefix=True,
    )

    begin_epoch = 0

    ##########
    resume_flag = os.path.exists(log_dir + '/saved_parameter.pth')
    if resume_flag:
        checkpoint = torch.load(log_dir + '/saved_parameter.pth')
        begin_epoch = checkpoint['epoch']+1
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best_result = checkpoint['max_accuracy']
        log_string('Model successfully restored from epoch %d' % begin_epoch)
    else:
        best_result = float(0)
    #############

    count = 0

    epoch_now = begin_epoch 
    best_result = validation(test_loader, net, loss_func, epoch_now, best_result, optimizer, epoch_now, lr_scheduler, log_string, log_dir)
    for i in range(begin_epoch, total_epoch):
        epoch_now = i
        count = train_one_epoch(train_loader, net, loss_func, loss_ce, loss_ie, optimizer, epoch_now, count, lr_scheduler, loss_meter, acc_meter)
        best_result = validation(test_loader, net, loss_func, epoch_now, best_result, optimizer, epoch_now, lr_scheduler, log_string, log_dir)
        functions.save_checkpoint(args, epoch_now, net, best_result, optimizer, lr_scheduler, log_string, log_dir)

def train_one_epoch(train_loader, net, loss_func, loss_ce, loss_ie, optimizer, epoch, count, lr_scheduler, loss_meter, acc_meter):
    log_string('Training at Epoch %d ---------------------------------' %(epoch))
    train_num_steps = len(train_loader)
    net.train()
    for idx, data in enumerate(train_loader):

        imgs, gt = data

        imgs = Variable(imgs).float().cuda()
        gt = Variable(gt).type(torch.cuda.LongTensor).cuda()

        pred, outputs1, outputs2, outputs3, pred_gt, outputs2_gt, outputs3_gt, diff_loss1, diff_loss2, diff_loss3 = net(imgs)

        loss11 = loss_func(pred, gt)

        loss1 = loss_ce(outputs3, pred_gt)
        loss2 = loss_ce(outputs2, outputs3_gt)
        loss3 = loss_ce(outputs1, outputs2_gt)
        loss4 = loss_ie(pred)
        loss5 = loss_ie(outputs3)
        loss6 = loss_ie(outputs2)

        loss = loss11 + 0.5*loss1 + 0.5*loss2 + 0.5*loss3 - 0.5*loss4 - 0.5*loss5 - 0.5*loss6 + 0.5*diff_loss1 + 0.5*diff_loss2 + 0.5*diff_loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), gt.shape[0])

        np_pred = pred.data.cpu().numpy()
        label = gt.data.cpu().numpy()
        pd = np.argmax(np_pred, axis=1)

        correct_num = np.sum(np.equal(pd, label).astype(np.uint8))

        acc = correct_num / float(label.shape[0]) * 100
        acc_meter.update(acc, gt.shape[0])

        if idx % 100 == 0:
            log_string('[Current epoch %d/%d, iter %d/%d, class loss is %2.5f(%2.5f), Training Top 1 Acc : %0.2f(%0.2f), lr: %f]'
                %(epoch, total_epoch, idx, train_num_steps, loss_meter.val, loss_meter.avg, acc_meter.val, acc_meter.avg, optimizer.param_groups[0]['lr']))

        count = count + 1
        lr_scheduler.step_update((epoch * train_num_steps + idx) // 1)

    return count


def validation(test_loader, net, loss_func, epoch, best_result, optimizer, epoch_now, lr_scheduler, log_string, log_dir):

    val_loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    all_preds = []
    all_gts = []

    with torch.no_grad():
        log_string('Validating at Epoch %d ---------------------------------' %(epoch))
        net.eval()
        test_loss_all = []
        test_correct = 0
        save_model = 0
        for idx, data in enumerate(test_loader):
            test_img, test_gt = data
            test_img = Variable(test_img).float().cuda(non_blocking=True)
            test_gt = Variable(test_gt).type(torch.cuda.LongTensor).cuda(non_blocking=True)
            test_pred, outputs1, outputs2, outputs3, pred_gt, outputs2_gt, outputs3_gt, diff_loss1, diff_loss2, diff_loss3 = net(test_img)
            softmax_pred = F.softmax(test_pred, dim=-1)
            test_loss = loss_func(test_pred, test_gt)


            _, preds = torch.max(test_pred, 1) ##test_pred = [B, 1000]
            
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(test_gt.cpu().numpy()) ## test_gt =[B]


            acc1, acc5 = accuracy(test_pred, test_gt, topk=(1, 5))

            val_loss_meter.update(test_loss.item(), test_gt.size(0))
            acc1_meter.update(acc1.item(), test_gt.size(0))
            acc5_meter.update(acc5.item(), test_gt.size(0))

            if idx % 10 == 0:
                log_string('[Test at epoch %d, iter %d/%d, class loss is %2.5f(%2.5f), Top 1 Acc : %0.2f(%0.2f)], Top 5 Acc : %0.2f(%0.2f)'
                    %(epoch, idx, len(test_loader), val_loss_meter.val, val_loss_meter.avg, acc1_meter.val, acc1_meter.avg, acc5_meter.val, acc5_meter.avg))

            _, predicted_labels = torch.max(softmax_pred, 1)

        all_preds = np.array(all_preds)
        all_gts = np.array(all_gts)

        AA_acc, kappa = calculate_metric(all_preds, all_gts, 1000)

        time_now = time.strftime('%Y-%m-%d.%H:%M:%S',time.localtime(time.time()))
        test_acc = acc1_meter.avg
        log_string('####################')
        log_string('The final overall accuracy is: {}'.format(test_acc))
        log_string('The final average accuracy is: {}'.format(AA_acc))
        log_string('The final Kappa score is: {}'.format(kappa))
        log_string('####################')
        if test_acc > best_result:
            best_result = test_acc
            save_model = 1
        log_string('The time now is' + time_now)
        log_string('[Class loss is %2.5f]' %(val_loss_meter.avg))
        log_string('[Testing Top 1 Acc : %0.2f]' %(test_acc))
        log_string('[Best Top 1 Acc : %0.2f]' %(best_result))
        if save_model == 1:
            log_string('[Get a new best result, save model now]')
            functions.save_checkpoint(args, epoch_now, net, best_result, optimizer, lr_scheduler, log_string, log_dir, best_flag=True)
        log_string('Validating Done, going back to Training')

        net.train()

        return best_result



if __name__ == "__main__":
	main()
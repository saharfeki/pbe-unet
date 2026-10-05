import os
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.dataset import MedicalDataSets
import albumentations as A
from utils.util import AverageMeter
import utils.losses_boundary as losses
from utils.metrics import iou_score
import csv
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from network.PBEUNet import PBEUNet



def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_torch(41)

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="PBEUNet",
                    choices=[""], help='model')
parser.add_argument('--base_dir', type=str, default="./data/busi", help='dir')
parser.add_argument('--train_file_dir', type=str, default="busi_train82.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="busi_val82.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
args = parser.parse_args()


def getDataloader():
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Resize(256, 256),
        A.Normalize(),
    ], additional_targets={'boundary': 'mask'})

    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
    ], additional_targets={'boundary': 'mask'})

    db_train = MedicalDataSets(base_dir=args.base_dir, split="train", transform=train_transform,
                               train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                             train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)

    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=8, shuffle=True,
                             num_workers=8, pin_memory=False)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=4)
    return trainloader, valloader


def get_model(args):  # 传入模型的
    if args.model == "PBEUNet":
        model = PBEUNet()
    else:
        model = None
        print("model err")
        exit(0)
    return model.cuda()


def main(args):
    base_lr = args.base_lr

    trainloader, valloader = getDataloader()
    model = get_model(args)
    print("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    criterion = losses.MultiTaskLoss().cuda()

    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    # best_dice = 0
    iter_num = 0
    max_epoch = 300

    max_iterations = len(trainloader) * max_epoch


    csv_dir = './metrics_log'  # 指定CSV文件存放的文件夹
    if not os.path.exists(csv_dir):  # 如果文件夹不存在，则创建文件夹d
        os.makedirs(csv_dir)
    csv_filename = os.path.join(csv_dir, 'metrics_PBEUNet_hunhe_BUSI_split82_{}.csv'.format(args.model))  # 文件名将是 training_metrics_CMunet.csv。
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Epoch', 'LR', 'train_loss', 'train_iou', 'train_dice', 'val_loss', 'val_iou',
                            'Val_dice', 'recall', 'precision', "F1", "specificity", "ACC", "HD95",
                            # "val_b5_loss", "val_b4_loss", "val_b3_loss", "val_b2_loss",
                            ])
        # writerow 方法将一行写入 CSV 文件，这一行包括了各列的表头。

    # 初始化余弦衰减学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-5)

    for epoch_num in range(max_epoch):
        model.train()
        avg_meters = {  # 这一段代码创建了一个字典avg_meters，用于存储训练和验证过程中的各种性能指标
            'train_loss': AverageMeter(),
            'train_iou': AverageMeter(),
            "train_dice": AverageMeter(),
            'val_loss': AverageMeter(),
            'val_iou': AverageMeter(),
            "val_dice": AverageMeter(),
            'recall': AverageMeter(),
            'precision': AverageMeter(),
            "specificity": AverageMeter(),
            'F1': AverageMeter(),
            'ACC': AverageMeter(),

            'HD95':AverageMeter(),

            # 'val_b5_loss': AverageMeter(),
            # 'val_b4_loss': AverageMeter(),
            # 'val_b3_loss': AverageMeter(),
            # 'val_b2_loss': AverageMeter(),

        }
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            boundary_batch = sampled_batch['boundary'].cuda()
            outputs, boundaries = model(volume_batch)
            loss, b_look = criterion(outputs, boundaries, label_batch, boundary_batch)
            iou, dice, _, _, _, _, _, _ = iou_score(outputs, label_batch)  # 计算指标
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # this


            for param_group in optimizer.param_groups:  # 遍历优化器中所有的参数组（param_groups）来更新每个参数组的学习率
                param_group['lr'] = lr_  # 当前参数组（param_group）的学习率被更新为变量 lr_ 的值

            iter_num += 1  # 迭代次数加1
            avg_meters['train_loss'].update(loss.item(), volume_batch.size(0))  # 更新各种指标的平均值
            avg_meters['train_iou'].update(iou, volume_batch.size(0))
            avg_meters['train_dice'].update(dice, volume_batch.size(0))

        # 在每个epoch结束后更新学习率
        scheduler.step()
        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input = sampled_batch['image'].cuda()
                target = sampled_batch['label'].cuda()
                boundary = sampled_batch['boundary'].cuda()
                output, boundaries = model(input)
                loss, b_look = criterion(output, boundaries, target, boundary)
                iou, dice, recall, precision, F1, specificity, acc, hd95 = iou_score(output, target)
                avg_meters['val_loss'].update(loss.item(), input.size(0))
                avg_meters['val_iou'].update(iou, input.size(0))
                avg_meters['val_dice'].update(dice, input.size(0))
                avg_meters['recall'].update(recall, input.size(0))
                avg_meters['precision'].update(precision, input.size(0))
                avg_meters['specificity'].update(specificity, input.size(0))
                avg_meters['F1'].update(F1, input.size(0))
                avg_meters['ACC'].update(acc, input.size(0))
                avg_meters['HD95'].update(hd95, input.size(0))

                # avg_meters['val_b5_loss'].update(b5.item(), input.size(0))
                # avg_meters['val_b4_loss'].update(b4.item(), input.size(0))
                # avg_meters['val_b3_loss'].update(b3.item(), input.size(0))
                # avg_meters['val_b2_loss'].update(b2.item(), input.size(0))


        # 将每个epoch的结果写入CSV文件
        with open(csv_filename, 'a', newline='') as csvfile:  # 使用with open()语句打开一个CSV文件，文件名为csv_filename。
            csvwriter = csv.writer(csvfile)  # 使用csv.writer()创建一个CSV写入器对象csvwriter，使得后续可以通过该对象将数据写入打开的CSV文件。
            csvwriter.writerow([  # csvwriter.writerow()方法写入一行数据到CSV文件
                epoch_num + 1,
                lr_,
                avg_meters['train_loss'].avg,
                avg_meters['train_iou'].avg,
                avg_meters['train_dice'].avg,
                avg_meters['val_loss'].avg,
                avg_meters['val_iou'].avg,
                avg_meters['val_dice'].avg,
                avg_meters['recall'].avg,
                avg_meters['precision'].avg,
                avg_meters['F1'].avg,
                avg_meters['specificity'].avg,
                avg_meters['ACC'].avg,
                avg_meters['HD95'].avg,

                # avg_meters['val_b5_loss'].avg,
                # avg_meters['val_b4_loss'].avg,
                # avg_meters['val_b3_loss'].avg,
                # avg_meters['val_b2_loss'].avg,

            ])

        print(
            'epoch [%d/%d], lr: %.6f, train_loss: %.4f, train_iou: %.4f, train_dice: %.4f, '
            '- val_loss: %.4f, val_iou: %.4f, val_dice: %.4f, recall: %.4f, precision: %.4f, '
            'F1: %.4f, specificity: %.4f, ACC: %.4f, HD95: %.4f, '
            # 'val_b5_loss: %.4f,val_b4_loss: %.4f, val_b3_loss: %.4f, val_b2_loss: %.4f,'
            % (epoch_num + 1, max_epoch, lr_, avg_meters['train_loss'].avg, avg_meters['train_iou'].avg,
               avg_meters['train_dice'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_dice'].avg,
               avg_meters['recall'].avg, avg_meters['precision'].avg, avg_meters['F1'].avg,
               avg_meters['specificity'].avg, avg_meters['ACC'].avg, avg_meters['HD95'].avg,
              # avg_meters['val_b5_loss'].avg, avg_meters['val_b4_loss'].avg,avg_meters['val_b3_loss'].avg,avg_meters['val_b2_loss'].avg,
               ))

        if avg_meters['val_iou'].avg > best_iou:
        # if avg_meters['val_dice'].avg > best_dice:
            if not os.path.exists('./checkpoint'):
                os.mkdir('checkpoint')  # 如果 checkpoint 目录不存在，使用 os.mkdir 创建该目录，以存放模型的保存文件。
            torch.save(model.state_dict(),
                       './checkpoint/{}_PBEUNet_hunhe_BUSI_split82.pth'.format(args.model))
            # 文件名的格式为 'checkpoint/{}_model_{}.pth'，其中 {} 部分由 args.model 替代，表示模型的名称，args.train_file_dir.split(".")[0]

            best_iou = avg_meters['val_iou'].avg
            # best_dice = avg_meters['val_dice'].avg
            print("=> saved best model")

        if epoch_num == max_epoch - 1:
            if not os.path.exists('./checkpoint'):
                os.mkdir('checkpoint')  # 如果 checkpoint 目录不存在，使用 os.mkdir 创建该目录，以存放模型的保存文件。
            torch.save(model.state_dict(),
                       './checkpoint/last_{}_PBEUNet_hunhe_BUSI_split82.pth'.format(args.model))
            # 文件名的格式为 'checkpoint/{}_model_{}.pth'，其中 {} 部分由 args.model 替代，表示模型的名称，args.train_file_dir.split(".")[0]
            print("=> saved last model")

    return "Training Finished!"


if __name__ == "__main__":
    main(args)



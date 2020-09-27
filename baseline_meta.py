import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from tensorboardX import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from dice_loss import dice_coeff

import matplotlib
import matplotlib.pyplot as plt
import IPython
import gc
import higher

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5, 
              img_size=512, 
              noise_fraction=0):

    dir_img = 'ISIC-2017_Training_Data/'
    dir_mask = 'ISIC-2017_Training_Part1_GroundTruth'
    dir_val_img = 'ISIC-2017_Training_Data_validation/'
    dir_val_mask = 'ISIC-2017_Training_Part1_GroundTruth_validation/'
    dir_cle_img = 'ISIC-2017_Training_Data_clean/'
    dir_cle_mask = 'ISIC-2017_Training_Part1_GroundTruth_validation_clean/'
    dir_checkpoint = 'checkpoints/'

    if noise_fraction != 0:
        dir_mask = dir_mask + '_' + str(noise_fraction) + '/'
        print(dir_mask)
    else:
        dir_mask = dir_mask + '/'
        print(dir_mask)

    train = BasicDataset(dir_img, dir_mask, img_scale, img_size)
    val = BasicDataset(dir_val_img, dir_val_mask, img_scale, img_size)
    cle = BasicDataset(dir_cle_img, dir_cle_mask, img_scale, img_size)
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    cle_loader = DataLoader(cle, batch_size=5, shuffle=False, num_workers=8, pin_memory=True)

    batch = next(iter(cle_loader))
    clean_data = batch['image']
    clean_labels = batch['mask']
    clean_data = clean_data.to(device=device, dtype=torch.float32)
    clean_labels = clean_labels.to(device=device, dtype=torch.float32)
    # clean_data = clean_data.cuda()
    # clean_labels = clean_labels.cuda()

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    net_losses = []
    acc_test = []
    acc_train = []
    dice_train = []
    dice_test = []
    loss_train = []
    num_batch = len(train_loader)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(train)}
        Validation size: {len(val)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Images size:     {img_size}
        Noise fraction:  {noise_fraction}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=0, momentum=0.99)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(epochs):
        net.train()
        tot = 0
        num_val = 0
        tot_val = 0
        epoch_loss = 0
        with tqdm(total=len(train), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                with higher.innerloop_ctx(net, optimizer) as (meta_net, meta_opt):
                    y_f_hat = meta_net(imgs)
                    loss = criterion(y_f_hat, true_masks)
                    eps = torch.zeros(cost.size()).cuda()
                    eps = eps.requires_grad_()
                    l_f_meta = torch.sum(cost * eps)
                    meta_opt.step(l_f_meta)

                    y_g_hat = meta_net(clean_data)
                    l_g_meta = torch.mean(criterion(y_g_hat, clean_labels))
                    grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True)[0].detach()

                w_tild = torch.clamp(-grad_eps, min=0)
                norm_c = torch.sum(w_tild)

                if norm_c != 0:
                    w = w_tild / norm_c
                else:
                    w = w_tild

                masks_pred = net(imgs)
                pred = torch.sigmoid(masks_pred)
                pred = (pred > 0.5).float()
                # print(pred.size())
                # print(true_masks[:, 0:1].size())
                tot += dice_coeff(pred, true_masks[:, 0:1]).item()
                dice_train.append(dice_coeff(pred, true_masks[:, 0:1]).item())
                writer.add_scalar('Dice/train', dice_coeff(pred, true_masks[:, 0:1]).item(), global_step)

                if dice_coeff(pred, true_masks[:, 0:1]).item() <= 0.3:
                    writer.add_images('masks/true', true_masks[:, 0:1], global_step)
                    writer.add_images('masks/pred', pred, global_step)

                cost = criterion(masks_pred, true_masks[:, 0:1])
                loss = torch.sum(cost * w)
                epoch_loss += loss.item()
                net_losses.append(loss.item())
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % (len(train) // (10 * batch_size)) == 0:
                    num_val += 1
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    dice_test.append(val_score)
                    tot_val += val_score
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        print('Step Validation Dice: ', val_score)
                        writer.add_scalar('Dice/test', val_score, global_step)

                    # writer.add_images('images', imgs, global_step)
                    # if net.n_classes == 1:
                    #     writer.add_images('masks/true', true_masks, global_step)
                    #     writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        print('Epoch: ', epoch)
        print('Epoch Loss: ', epoch_loss/num_batch)
        loss_train.append(epoch_loss/num_batch)

        print('Train EpochDice: ', tot/num_batch)
        acc_train.append(tot/num_batch)
        writer.add_scalar('EpochDice/train', tot/num_batch, epoch)

        print('Val EpochDice: ', tot_val/num_val)
        acc_test.append(tot_val/num_val)
        writer.add_scalar('EpochDice/test', tot_val/num_val, epoch)
        
        path = dir_checkpoint + args.figpath + '_model.pth'
        # path = 'baseline/' + str(args.noise_fraction) + '/model.pth'
        torch.save(net.state_dict(), path)
    
    IPython.display.clear_output()
    fig, axes = plt.subplots(3, 2, figsize=(13, 5))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    ax1.plot(net_losses, label='iteration_losses')
    ax1.set_ylabel("Losses")
    ax1.set_xlabel("Iteration")
    ax1.legend()

    ax2.plot(loss_train, label='epoch_losses')
    ax2.set_ylabel('Losses')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    ax3.plot(acc_train, label='dice_train_epoch')
    ax3.set_ylabel('EpochDice/train')
    ax3.set_xlabel('Epoch')
    ax3.legend()

    ax4.plot(acc_test, label='dice_test_epoch')
    ax4.set_ylabel('EpochDice/test')
    ax4.set_xlabel('Epoch')
    ax4.legend()

    ax5.plot(dice_train, label='dice_train_iteration')
    ax5.set_ylabel('IterationDice/train')
    ax5.set_xlabel('Iteration')
    ax5.legend()

    ax6.plot(dice_test, label='dice_test_iteration')
    ax6.set_ylabel('IterationDice/test')
    ax6.set_xlabel('Iteration')
    ax6.legend()

    plt.savefig(args.figpath+'.png')

    writer.close()
    return net


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-c', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-s', '--size', dest='size', type=int, default=512,
                        help='Size of images')
    parser.add_argument('-n', '--noise-fraction', metavar='NF', type=float, nargs='?', default=0.2,
                        help='Noise Fraction', dest='noise_fraction')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-p', '--fig-path', metavar='FP', type=str, nargs='?', default='baseline',
                        help='Fig Path', dest='figpath')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    # net = UNet(n_channels=3, n_classes=1, bilinear=True)
    net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana')
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        net = train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  img_size=args.size, 
                  val_percent=args.val/100, 
                  noise_fraction=args.noise_fraction)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

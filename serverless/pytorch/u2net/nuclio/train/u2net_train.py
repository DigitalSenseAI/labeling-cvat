import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import sys

from try_u2net.custom_logging.logger import logger
from try_u2net.custom_logging.logger import setup_logger

from u2net.u2net.data_loader import Rescale
from u2net.u2net.data_loader import RescaleT
from u2net.u2net.data_loader import RandomCrop
from u2net.u2net.data_loader import ToTensor
from u2net.u2net.data_loader import ToTensorLab
from u2net.u2net.data_loader import SalObjDataset

from u2net.u2net.model import U2NET
from u2net.u2net.model import U2NETP
from sklearn.model_selection import train_test_split

TRAIN_CONFIG_PATH = 'train/config.yaml'
MODEL_NAME = 'u2net'

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss = nn.BCELoss(size_average=True)

    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss

def get_model(model_name):
    if model_name == 'u2net':
        net = U2NET(3, 1)
    elif model_name == 'u2netp':
        net = U2NETP(3, 1)
    return net.cuda() if torch.cuda.is_available() else net

def get_config():
    with open(TRAIN_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_dataset(config):
    data_cfg = config['dataset']
    # tra_image_dir = Path('DUTS') / Path('DUTS-TR') / Path('DUTS-TR') / 'im_aug'
    # tra_label_dir = Path('DUTS') / Path('DUTS-TR') / Path('DUTS-TR') / 'gt_aug'  # TODO: augmentation?

    dataset_dir = Path(data_cfg['dir_path'])

    images = sorted(
        str(p) for p in (dataset_dir / 'train').rglob('*')
        if p.is_file()
    )
    labels = sorted(
        str(p) for p in (dataset_dir / 'labels_3').rglob('*') 
        if p.is_file()
    )
    
    # Combine images and labels into pairs
    data_pairs = list(zip(images, labels))

    # Split into train and validation sets (90% train, 10% validation)
    train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.1, random_state=42)

    # Unzip the pairs back into separate lists
    tra_img_name_list, tra_lbl_name_list = zip(*train_pairs)
    val_img_name_list, val_lbl_name_list = zip(*val_pairs)

    # Convert tuples back to lists
    tra_img_name_list = list(tra_img_name_list)
    tra_lbl_name_list = list(tra_lbl_name_list)
    val_img_name_list = list(val_img_name_list)
    val_lbl_name_list = list(val_lbl_name_list)

    assert set(tra_img_name_list).isdisjoint(set(val_img_name_list)), 'Train and validation images overlap'
    assert set([Path(p).stem for p in tra_img_name_list]) == set([Path(p).stem for p in tra_lbl_name_list]), 'Train images and labels do not match'
    assert set([Path(p).stem for p in val_img_name_list]) == set([Path(p).stem for p in val_lbl_name_list]), 'Validation images and labels do not match'

    assert all(Path(p).exists() for p in tra_img_name_list), 'Some train images do not exist'
    assert all(Path(p).exists() for p in tra_lbl_name_list), 'Some train labels do not exist'
    assert all(Path(p).exists() for p in val_img_name_list), 'Some val images do not exist'
    assert all(Path(p).exists() for p in val_lbl_name_list), 'Some val labels do not exist'

    return tra_img_name_list, tra_lbl_name_list, val_img_name_list, val_lbl_name_list

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            inputs = inputs.to(torch.float)
            labels = labels.to(torch.float)
            d0, d1, d2, d3, d4, d5, d6 = model(inputs)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss

def plot_losses(output_dir, losses):
    plt.figure()
    plt.plot(losses['train'], label='Train Loss')
    plt.plot(losses['val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'losses.png')
    plt.close()

def main(config):
    model_name = 'u2net' #'u2netp'

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = Path(config['output_model_dir']) / f"{model_name}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    log_file = model_dir / 'training.log'
    setup_logger(log_file)

    logger.info(f'configuration: {config}')

    data_cfg = config['dataset']
    # tra_image_dir = Path('DUTS') / Path('DUTS-TR') / Path('DUTS-TR') / 'im_aug'  # TODO: augmentation?
    # tra_label_dir = Path('DUTS') / Path('DUTS-TR') / Path('DUTS-TR') / 'gt_aug'

    epoch_num = config['train']['epoch_num']
    train_batch_size = config['train']['batch_size']

    train_num = 0
    val_num = 0

    tra_img_name_list, tra_lbl_name_list, val_img_name_list, val_lbl_name_list = get_dataset(config)

    logger.info(f"train images: {len(tra_img_name_list)}")
    logger.info(f"train labels: {len(tra_lbl_name_list)}")
    logger.info(f"val images: {len(val_img_name_list)}")
    logger.info(f"val labels: {len(val_lbl_name_list)}")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1)

    val_dataset = SalObjDataset(
        img_name_list=val_img_name_list,
        lbl_name_list=val_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            ToTensorLab(flag=0)]))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    net = get_model(model_name)

    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    logger.info('Starting training')
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq_in_ep = config['train']['save_frq_in_ep']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = {'train': [], 'val': []}

    best_val_loss = float('inf')

    for epoch in range(0, epoch_num):
        net.train()
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0
        train_epoch_loss = 0

        for i, data in enumerate(salobj_dataloader):
            ite_num += 1
            ite_num4val += 1

            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            inputs_v, labels_v = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device), requires_grad=False)

            optimizer.zero_grad()
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()
            train_epoch_loss += loss.data.item()

            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            logger.info("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * train_batch_size, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if (epoch + 1) % save_frq_in_ep == 0:
            torch.save(net.state_dict(), model_dir / f"{model_name}_epoch_{epoch + 1}_train_{train_epoch_loss:.3f}_tar_{running_tar_loss / ite_num4val:.3f}.pth")

        train_epoch_loss /= len(salobj_dataloader)

        val_epoch_loss = validate_model(net, val_dataloader, muti_bce_loss_fusion, device)
        losses['train'].append(train_epoch_loss)
        losses['val'].append(val_epoch_loss)
        logger.info(f'Epoch {epoch}/{epoch_num - 1}, Train Loss: {train_epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}')

        plot_losses(model_dir, losses)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(net.state_dict(), model_dir / f"{model_name}_best_val_model.pth")
            logger.info(f'Saving model with Validation Loss: {val_epoch_loss:.4f}')

if __name__ == '__main__':
    config = get_config()
    main(config)

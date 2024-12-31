import os
import sys
import torch
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data as data
import torch.nn.functional as F
from Dataset.init_dataset import Cifar10FL,Cifar100FL,Cifar_C,CelebA,ImageFolder_custom
from torch.autograd import Variable
from Network.Models_Def.resnet import ResNet10,ResNet12,ResNet18
from Network.Models_Def.shufflenet import ShuffleNetG2
from Network.Models_Def.mobilnet_v2 import MobileNetV2
import torchvision.transforms as transforms
from torchvision import datasets
import random

Seed = 0
seed = Seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
Project_Path = r'/home/fangxiuwen/'

def init_logs(log_level=logging.INFO,log_path = Project_Path+'Logs/',sub_name=None):
    # logging：https://www.cnblogs.com/CJOKER/p/8295272.html
    # 第一步，创建一个logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    log_path = log_path
    mkdirs(log_path)
    filename = os.path.basename(sys.argv[0][0:-3])
    if sub_name == None:
        log_name = log_path + filename + '.log'
    else:
        log_name = log_path + filename + '_' + sub_name +'.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(log_level)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    console  = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(console)
    # 日志
    return logger

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_cifar10_data(datadir, noise_type=None, noise_rate=0):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_train_ds = Cifar10FL(datadir, train=True, download=True, transform=transform, noise_type=noise_type, noise_rate=noise_rate)
    cifar10_test_ds = Cifar10FL(datadir, train=False, download=True, transform=transform)
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir, noise_type=None, noise_rate=0):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar100_train_ds = Cifar100FL(datadir, train=True, download=True, transform=transform, noise_type=noise_type, noise_rate=noise_rate)
    cifar100_test_ds = Cifar100FL(datadir, train=False, download=True, transform=transform)
    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_celeba_data(datadir, noise_type=None, noise_rate=0):
    transform = transforms.Compose([transforms.ToTensor()])
    celeba_train_ds = CelebA(datadir, train=True, download=True, transform=transform)
    celeba_test_ds = CelebA(datadir, train=False, download=True, transform=transform)
    X_train, y_train = celeba_train_ds.data, celeba_train_ds.target
    X_test, y_test = celeba_test_ds.data, celeba_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_tiny_imagenet_data(datadir, noise_type=None, noise_rate=0):
    transform = transforms.Compose([transforms.ToTensor()])
    tiny_imagenet_train_ds = ImageFolder_custom(datadir, train=True, download=True, transform=transform)
    tiny_imagenet_test_ds = ImageFolder_custom(datadir, train=False, download=True, transform=transform)
    X_train, y_train = tiny_imagenet_train_ds.data, tiny_imagenet_train_ds.target
    X_test, y_test = tiny_imagenet_test_ds.data, tiny_imagenet_test_ds.target
    return (X_train, y_train, X_test, y_test)

def partition_data(dataset, datadir, partition, n_parties, logger, beta=1.0, noise_type=None, noise_rate=0):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir, noise_type=noise_type, noise_rate=noise_rate)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir, noise_type=noise_type, noise_rate=noise_rate)
    n_train = y_train.shape[0]
    if partition == 'iid':
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_splits(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
    elif partition == "noniid":
        min_size = 1
        min_require_size = 10
        K = 10 # Class Number
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tiny_imagenet':
            K = 200
            # min_require_size = 100
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0] # get the corresponding sample index if label == k
                np.random.shuffle(idx_k) # Shuffle sample index
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    # traindata_cls_counts = _record_net_data_stats(y_train, net_dataidx_map, logger)
    return net_dataidx_map
    # return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def generate_public_data_indexs(dataset,datadir,size, noise_type=None, noise_rate=0):
    if dataset =='cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir, noise_type=noise_type, noise_rate=noise_rate)
    if dataset =='cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir, noise_type=noise_type, noise_rate=noise_rate)
    if dataset =='celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir, noise_type=noise_type, noise_rate=noise_rate)
    if dataset =='tiny_imagenet':
        X_train, y_train, X_test, y_test = load_tiny_imagenet_data(datadir, noise_type=noise_type, noise_rate=noise_rate)
    if size <= 10000:
        n_train = y_train.shape[0]
    else:
        n_train = X_train.shape[0]
    idxs = np.random.permutation(n_train)
    idxs = idxs[0:size]
    return idxs

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, noise_type=None, noise_rate=0):
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = Cifar10FL
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        if dataset =='cifar100':
            dl_obj=Cifar100FL
            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        # noise_type = 'pairflip'  # [pairflip, symmetric]
        # noise_rate = 0.1
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=False, noise_type=noise_type, noise_rate=noise_rate)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=False)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)
    #
    #
    if dataset == 'celeba':
        normalize = transforms.Normalize(mean=[0.50612009, 0.42543493, 0.38282761], std=[0.26589054, 0.24521921, 0.24127836])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if dataidxs is None:
            train_ds = datasets.ImageFolder(datadir+'/img', transform=transform_train)
        else:
            train_ds = datasets.ImageFolder(datadir+'/img', transform=transform_train)
            train_ds = data.Subset(train_ds, dataidxs)
        test_ds = datasets.ImageFolder(datadir+'/img', transform=transform_test)
        train_dl = data.DataLoader(dataset=train_ds, drop_last=True, batch_size=train_bs, shuffle=True, num_workers=4)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    if dataset == 'tiny_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if dataidxs is None:
            train_ds = datasets.ImageFolder(datadir+'/train', transform=transform_train)
        else:
            train_ds = datasets.ImageFolder(datadir+'/train', transform=transform_train)
            train_ds = data.Subset(train_ds, dataidxs)
        test_ds = datasets.ImageFolder(datadir+'/test', transform=transform_test)
        train_dl = data.DataLoader(dataset=train_ds, drop_last=True, batch_size=train_bs, shuffle=True, num_workers=4)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)
    return train_dl, test_dl, train_ds, test_ds

def init_nets(n_parties,nets_name_list,num_classes):
    nets_list = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net_name = nets_name_list[net_i]
        if net_name=='ResNet10':
            net = ResNet10(num_classes=num_classes)
        elif net_name =='ResNet12':
            net = ResNet12(num_classes=num_classes)
        elif net_name =='ShuffleNet':
            net = ShuffleNetG2(num_classes=num_classes)
        elif net_name =='Mobilenetv2':
            net = MobileNetV2(num_classes=num_classes)
        elif net_name =='ResNet18':
            net = ResNet18(num_classes=num_classes)
        nets_list[net_i] = net
    return nets_list

if __name__ =='__main__':
    logger = init_logs()
    net_dataidx_map = partition_data(dataset='cifar10',datadir='./cifar_10',partition='noniid', n_parties=10,logger=logger)
    public_data_indexs = generate_public_data_indexs(dataset='cifar100',datadir='./cifar_100',size=5000)
    train_dl, test_dl, train_ds, test_ds = get_dataloader(dataset='cifar100',datadir='./cifar_100',train_bs=256,test_bs=512,dataidxs=public_data_indexs)
    print(len(train_ds))
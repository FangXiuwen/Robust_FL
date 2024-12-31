import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
sys.path.append("..")
from Dataset.utils import init_logs, partition_data, get_dataloader, init_nets, mkdirs
from Network.Models_Def.resnet import ResNet10,ResNet12
from matplotlib.colors import ListedColormap
from loss import SCELoss, AGCELoss, NCEandAGCE
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import manifold
import torch.nn as nn
import numpy as np
from numpy import *
import random
import torch

Noniid_beta = None #[data homo:None, hetero:0.5]

Noise_type = 'symmetric' #['pairflip','symmetric','None']
Noise_rate = 0.2
Seed = 0
N_Participants = 4
TrainBatchSize = 256
TestBatchSize = 512
Pretrain_Epoch = 40
Private_Data_Len = 10000

Pariticpant_Params = {
    'loss_funnction' : 'SCE',
    'optimizer_name' : 'Adam',
    'learning_rate'  : 0.001
}
Sce_alpha = 0.4
Sce_beta = 0.9
# Nets_Name_List = ['ResNet10','ResNet12','ShuffleNet','Mobilenetv2','ResNet10','ResNet10','ResNet10','ResNet10','ResNet10','ResNet10']
Nets_Name_List = ['ResNet10','ResNet12','ShuffleNet','Mobilenetv2']
Dataset_Name = 'cifar10'
Dataset_Dir = '../Dataset/cifar_10'
Dataset_Classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Output_Channel = len(Dataset_Classes)

def pretrain_network(epoch,net,data_loader,loss_function,optimizer_name,learning_rate):
    if loss_function =='CE':
        criterion = nn.CrossEntropyLoss()
    if loss_function =='SCE':
        criterion = SCELoss(alpha=Sce_alpha, beta=Sce_beta, num_classes=10)
    criterion.to(device)
    if optimizer_name =='Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    if optimizer_name =='SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    net.train()
    for _epoch in range(epoch):
        log_interval = 100
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    _epoch, batch_idx * len(images), len(data_loader.dataset),
                           100. * batch_idx / len(data_loader), loss.item()))
    return net

def evaluate_network(net,dataloader):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return 100 * correct / total


if __name__ =='__main__':
    mkdirs('./Model_Storage/beta' + str(Noniid_beta) + '/' + Pariticpant_Params['loss_funnction'] + '/' + str(Noise_type) +str(Noise_rate))
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    seed = Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_ids = [0,1,2,3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger.info("Load Participants' Data and Model")
    if Noniid_beta == None:
        net_dataidx_map = {}
        for index in range(N_Participants):
            idxes = np.random.permutation(50000)
            idxes = idxes[0:Private_Data_Len]
            net_dataidx_map[index]= idxes
    else:
        net_dataidx_map = partition_data(dataset=Dataset_Name,datadir=Dataset_Dir,partition='noniid', n_parties=N_Participants,logger=logger,beta=Noniid_beta)
    logger.info(net_dataidx_map)
    net_list = init_nets(n_parties=N_Participants,nets_name_list=Nets_Name_List,num_classes=Output_Channel)

    logger.info('Pretrain Participants Models')
    for index in range(N_Participants):
        train_dl_local ,test_dl, train_ds_local, test_ds = get_dataloader(dataset=Dataset_Name,datadir=Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,dataidxs=net_dataidx_map[index], noise_type=Noise_type, noise_rate=Noise_rate)
        network = net_list[index]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Nets_Name_List[index]
        logger.info('Pretrain the '+str(index)+' th Participant Model with N_training: '+str(len(train_ds_local)))
        network = pretrain_network(epoch=Pretrain_Epoch,net=network,data_loader=train_dl_local,loss_function=Pariticpant_Params['loss_funnction'],optimizer_name=Pariticpant_Params['optimizer_name'],learning_rate=Pariticpant_Params['learning_rate'])
        logger.info('Save the '+str(index)+' th Participant Model')
        torch.save(network.state_dict(), './Model_Storage/beta'+ str(Noniid_beta) + '/' +Pariticpant_Params['loss_funnction'] + '/' + str(Noise_type) + str(Noise_rate)+ '/'+netname+'_'+str(index)+'.ckpt')
    logger.info('Evaluate Models')
    test_accuracy_list = []
    for index in range(N_Participants):
        _ ,test_dl, _, _ = get_dataloader(dataset=Dataset_Name,datadir=Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,dataidxs=net_dataidx_map[index])
        network = net_list[index]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Nets_Name_List[index]
        network.load_state_dict(torch.load('./Model_Storage/beta'+ str(Noniid_beta) + '/'+Pariticpant_Params['loss_funnction'] + '/' + str(Noise_type) + str(Noise_rate)+ '/'+netname+'_'+str(index)+'.ckpt'))
        output = evaluate_network(net=network,dataloader=test_dl)
        test_accuracy_list.append(output)
    print('The average Accuracy of models on the test images:'+str(mean(test_accuracy_list)))

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from copy import deepcopy
import sys
sys.path.append("..")
from Dataset.utils import init_logs, partition_data, get_dataloader, init_nets, generate_public_data_indexs, mkdirs
from loss import SCELoss,SelfAdaptiveTrainingSCE
import torch.nn.functional as F
import torch.optim as optim
from random import sample
import math
import torch.nn as nn
import numpy as np
from numpy import *
import random
import torch
import torch.backends.cudnn
import os
from tensorboardX import SummaryWriter

'''
Global Parameters
'''
Noniid_beta = None #[data homo:None, hetero:0.5]

Noise_type = 'pairflip' #['pairflip','symmetric','None']
Noise_rate = 0.2

Seed = 0
N_Participants = 4 #10
TrainBatchSize = 256 # 256
TestBatchSize = 512
CommunicationEpoch = 40
Private_Data_Len = 10000
Noise_Robust_Aggregation = True
Noise_Robust_Aggregation_Loss = 'SCE'
if Noise_Robust_Aggregation == False:
    beta = 0
else:
    beta = 1.2  #1.4
Tempreture = 4.0 #4.0
Sat_ratio = 10.0 #10.0 

Pariticpant_Params = {
    'loss_funnction' : 'SCE', #CE, SCE
    'optimizer_name' : 'Adam',
    'learning_rate'  : 0.001
}
Sce_alpha = 0.4
Sce_beta = 0.9
Private_Nets_Name_List = ['ResNet10','ResNet12','ShuffleNet','Mobilenetv2']
# Private_Nets_Name_List = ['ResNet12','ResNet12','ResNet12','ResNet12'] #model homo
Private_Dataset_Name = 'cifar10'
Private_Dataset_Dir = '../Dataset/cifar_10'
Private_Dataset_Classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Private_Output_Channel = len(Private_Dataset_Classes)
Public_Dataset_Name = 'cifar100' #cifar100,celeba,tiny_imagenet
Public_Dataset_Dir = '../Dataset/cifar_100'
# Public_Dataset_Dir = '../Dataset/celeba'
Public_Dataset_Length = 5000

def evaluate_network(network,dataloader,logger):
    network.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        logger.info('Test Accuracy of the model on the test images: {} %'.format(acc))
    return acc


def update_model_via_private_data(network,private_epoch,private_dataloader,loss_function,optimizer_method,learing_rate,logger,com_epoch):
    if loss_function =='CE':
        criterion = nn.CrossEntropyLoss()
    if loss_function =='SCE':
        criterion = SelfAdaptiveTrainingSCE(num_classes=10, alpha=Sce_alpha, beta=Sce_beta, temp=Tempreture, mu=Sat_ratio, com_epoch=com_epoch, total_epoch=CommunicationEpoch) #0.1,1

    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learing_rate)
    participant_local_loss_batch_list = []
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = network(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch_index % 5 ==0:
                logger.info('Private Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(images), len(private_dataloader.dataset),
                    100. * batch_idx / len(private_dataloader), loss.item()))
    return network,participant_local_loss_batch_list

if __name__ =='__main__':
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

    logger.info("Initialize Participants' Data idxs and Model")
    if Noniid_beta == None:
        net_dataidx_map = {}
        for index in range(N_Participants):
            idxes = np.random.permutation(50000)
            idxes = idxes[0:Private_Data_Len]
            net_dataidx_map[index]= idxes
    else:
        net_dataidx_map = partition_data(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,partition='noniid', n_parties=N_Participants,logger=logger,beta=Noniid_beta)
    logger.info(net_dataidx_map)

    net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Nets_Name_List,num_classes=Private_Output_Channel)
    logger.info("Load Participants' Models")
    for i in range(N_Participants):
        network = net_list[i]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Private_Nets_Name_List[i]
        network.load_state_dict(torch.load('../Network/Model_Storage/beta' + str(Noniid_beta) + '/' + Pariticpant_Params['loss_funnction'] + '/' + str(Noise_type) + str(Noise_rate)+ '/' + netname + '_' + str(i) + '.ckpt'))

    logger.info("Initialize Public Data Parameters")
    public_data_indexs = generate_public_data_indexs(dataset=Public_Dataset_Name,datadir=Public_Dataset_Dir,size=Public_Dataset_Length, noise_type=Noise_type, noise_rate=Noise_rate)
    public_train_dl, _, public_train_ds, _ = get_dataloader(dataset=Public_Dataset_Name,datadir=Public_Dataset_Dir,
    train_bs=TrainBatchSize,test_bs=TestBatchSize,dataidxs=public_data_indexs, noise_type=Noise_type, noise_rate=Noise_rate)

    col_loss_list = []
    local_loss_list = []
    acc_list = []
    current_mean_loss_list = [] #for reweight
    for epoch_index in range(CommunicationEpoch):
        logger.info("The "+str(epoch_index)+" th Communication Epoch")

        pre_col_network_list = []

        logger.info('Evaluate Models')
        acc_epoch_list = []
        for participant_index in range(N_Participants):
            netname = Private_Nets_Name_List[participant_index]
            private_dataset_dir = Private_Dataset_Dir
            _, test_dl, _, _ = get_dataloader(dataset=Private_Dataset_Name, datadir=private_dataset_dir,
                                              train_bs=TrainBatchSize,
                                              test_bs=TestBatchSize, dataidxs=None, noise_type=Noise_type, noise_rate=Noise_rate)
            network = net_list[participant_index]
            pre_col_network = deepcopy(network)
            pre_col_network_list.append(pre_col_network)
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
            acc_epoch_list.append(accuracy)
        acc_list.append(acc_epoch_list)
        accuracy_avg = sum(acc_epoch_list) / N_Participants

        # '''
        # wo ECCR/CCR
        # '''
        # weight_with_quality = [1 / (N_Participants - 1) for i in range(N_Participants)]
        # weight_with_quality = torch.tensor(weight_with_quality)

        '''
        Calculate quality with Data Quality Measurement
        '''
        amount_with_quality = [1 / (N_Participants - 1) for i in range(N_Participants)]
        weight_with_quality = [1 / (N_Participants - 1) for i in range(N_Participants)]
        quality_list = []
        last_mean_loss_list = current_mean_loss_list
        current_mean_loss_list = []

        for participant_index in range(N_Participants):
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            with torch.no_grad():
                network.eval()
                private_dataidx = net_dataidx_map[participant_index]
                train_dl_local, _, train_ds_local, _ = get_dataloader(dataset=Private_Dataset_Name, datadir=Private_Dataset_Dir,
                                                                    train_bs=TrainBatchSize, test_bs=TestBatchSize,
                                                                    dataidxs=private_dataidx, noise_type=Noise_type, noise_rate=Noise_rate)
                if Noise_Robust_Aggregation_Loss == 'CE':
                    criterion = nn.CrossEntropyLoss()
                if Noise_Robust_Aggregation_Loss == ('SCE' or 'DLRSCE'):
                    criterion = SCELoss(alpha=Sce_alpha, beta=Sce_beta, num_classes=10)
                criterion.to(device)
                participant_loss_list = []
                for batch_idx, (images, labels) in enumerate(train_dl_local):
                    images = images.to(device)
                    labels = labels.to(device)
                    private_linear_output, _ = network(images)
                    private_loss = criterion(private_linear_output, labels)
                    participant_loss_list.append(private_loss.item())
            mean_participant_loss = mean(participant_loss_list)
            current_mean_loss_list.append(mean_participant_loss)
        # EXP Standardized processing
        if epoch_index > 0 and beta != 0:
            abs_quality_list = []
            for participant_index in range(N_Participants):
                delta_loss = last_mean_loss_list[participant_index] - current_mean_loss_list[participant_index]
                delta_model = params_dif_list[participant_index]
                learning_efficiency = delta_loss / (delta_model + 1)
                participant_quality = learning_efficiency / current_mean_loss_list[participant_index]
                quality_list.append(participant_quality)
                abs_quality_list.append(abs(participant_quality))
            quality_sum = sum(abs_quality_list)
            for participant_index in range(N_Participants):
                amount_with_quality[participant_index] += beta * quality_list[participant_index] / quality_sum
            amount_with_quality_sum = sum(amount_with_quality)
            for participant_index in range(N_Participants):
                weight_with_quality[participant_index] = amount_with_quality[participant_index] / amount_with_quality_sum
        weight_with_quality = torch.tensor(weight_with_quality)

        # '''CCR'''
        # if epoch_index > 0 :
        #     for participant_index in range(N_Participants):
        #         delta_loss = last_mean_loss_list[participant_index] - current_mean_loss_list[participant_index]
        #         quality_list.append(delta_loss / current_mean_loss_list[participant_index])
        #     quality_sum = sum(quality_list)
        #     for participant_index in range(N_Participants):
        #         amount_with_quality[participant_index] += beta * quality_list[participant_index] / quality_sum
        #         # amount_with_quality_exp.append(exp(amount_with_quality[participant_index]))
        #     amount_with_quality_sum = sum(amount_with_quality)
        #     for participant_index in range(N_Participants):
        #         weight_with_quality.append(amount_with_quality[participant_index] / amount_with_quality_sum)
        # else:
        #     weight_with_quality = [1 / (N_Participants - 1) for i in range(N_Participants)]
        # weight_with_quality = torch.tensor(weight_with_quality)

        '''
        HHF
        '''
        for batch_idx, (images, _) in enumerate(public_train_dl):
            linear_output_list = []
            linear_output_target_list = []
            kl_loss_batch_list = []
            '''
            Calculate Linear Output
            '''
            for participant_index in range(N_Participants):
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network.train()
                images = images.to(device)
                linear_output,_ = network(x=images)
                linear_output_softmax = F.softmax(linear_output,dim =1)
                linear_output_target_list.append(linear_output_softmax.clone().detach())
                linear_output_logsoft = F.log_softmax(linear_output,dim=1)
                linear_output_list.append(linear_output_logsoft)
            '''
            Update Participants' Models via KL Loss and Data Quality
            '''
            for participant_index in range(N_Participants):
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network.train()
                criterion = nn.KLDivLoss(reduction='batchmean')
                criterion.to(device)
                optimizer = optim.Adam(network.parameters(), lr=Pariticpant_Params['learning_rate'])
                optimizer.zero_grad()
                loss = torch.tensor(0)
                for i in range(N_Participants):
                    if i != participant_index:
                        weight_index = weight_with_quality[i]
                        loss_batch_sample = criterion(linear_output_list[participant_index], linear_output_target_list[i])
                        temp = weight_index * loss_batch_sample
                        loss = loss + temp
                kl_loss_batch_list.append(loss.item())
                loss.backward()
                optimizer.step()
            col_loss_list.append(kl_loss_batch_list)


        '''
        Update Participants' Models via Private Data
        '''
        local_loss_batch_list = []
        params_dif_list = []
        for participant_index in range(N_Participants):
            pre_col_network = pre_col_network_list[participant_index]
            pre_col_network = nn.DataParallel(pre_col_network, device_ids=device_ids).to(device)
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            private_dataidx = net_dataidx_map[participant_index]
            train_dl_local, _, train_ds_local, _ = get_dataloader(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,
            train_bs=TrainBatchSize,test_bs=TestBatchSize,dataidxs=private_dataidx, noise_type=Noise_type, noise_rate=Noise_rate)
            private_epoch = max(int(len(public_train_ds)/len(train_ds_local)),1)
            network,private_loss_batch_list = update_model_via_private_data(network=network,private_epoch=private_epoch,
            private_dataloader=train_dl_local,loss_function=Pariticpant_Params['loss_funnction'],
            optimizer_method=Pariticpant_Params['optimizer_name'],learing_rate=Pariticpant_Params['learning_rate'],
            logger=logger, com_epoch=epoch_index)
            mean_privat_loss_batch = mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_privat_loss_batch)

            '''Calculate params_dif'''
            params_norm = 0.0
            params_count = 0
            for param_after, param_pre in zip(network.parameters(), pre_col_network.parameters()):
                params_norm += torch.norm(param_after - param_pre).item()
                params_count += 1
            params_dif_list.append(params_norm / params_count)

        local_loss_list.append(local_loss_batch_list)

        if epoch_index == CommunicationEpoch - 1:
            acc_epoch_list = []
            logger.info('Final Evaluate Models')
            for participant_index in range(N_Participants):  # 改成2 拿来测试 N_Participants
                _, test_dl, _, _ = get_dataloader(dataset=Private_Dataset_Name, datadir=Private_Dataset_Dir,
                                                  train_bs=TrainBatchSize,
                                                  test_bs=TestBatchSize, dataidxs=None, noise_type=Noise_type, noise_rate=Noise_rate)
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
                acc_epoch_list.append(accuracy)
            acc_list.append(acc_epoch_list)
            accuracy_avg = sum(acc_epoch_list) / N_Participants

        if epoch_index % 5 == 0 or epoch_index==CommunicationEpoch-1:
            mkdirs('./test/Performance_Analysis/'+Pariticpant_Params['loss_funnction'])
            mkdirs('./test/Model_Storage/' +Pariticpant_Params['loss_funnction'])
            mkdirs('./test/Performance_Analysis/'+Pariticpant_Params['loss_funnction']+str(Noise_type))
            mkdirs('./test/Model_Storage/'+Pariticpant_Params['loss_funnction']+str(Noise_type))
            mkdirs('./test/Performance_Analysis/'+Pariticpant_Params['loss_funnction']+'/'+str(Noise_type)+str(Noise_rate))
            mkdirs('./test/Model_Storage/' +Pariticpant_Params['loss_funnction']+'/'+str(Noise_type)+ str(Noise_rate))

            logger.info('Save Loss')
            col_loss_array = np.array(col_loss_list)
            np.save('./test/Performance_Analysis/' +Pariticpant_Params['loss_funnction']+'/'+ str(Noise_type) +str(Noise_rate)
                    +'/collaborative_loss.npy', col_loss_array)
            local_loss_array = np.array(local_loss_list)
            np.save('./test/Performance_Analysis/'+Pariticpant_Params['loss_funnction']+'/'+str(Noise_type) +str(Noise_rate)
                    +'/local_loss.npy', local_loss_array)
            logger.info('Save Acc')
            acc_array = np.array(acc_list)
            np.save('./test/Performance_Analysis/' +Pariticpant_Params['loss_funnction']+'/'+ str(Noise_type) +str(Noise_rate)
                    +'/acc.npy', acc_array)

            logger.info('Save Models')
            for participant_index in range(N_Participants):
                netname = Private_Nets_Name_List[participant_index]
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                torch.save(network.state_dict(),
                           './test/Model_Storage/' +Pariticpant_Params['loss_funnction']+'/'+ str(Noise_type) + str(Noise_rate) + '/'
                           + netname + '_' + str(participant_index) + '.ckpt')
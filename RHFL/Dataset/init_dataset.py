from Dataset.cifar import CIFAR10,CIFAR100
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

'''
Initialize Dataset
'''

class Cifar10FL(Dataset):
    def __init__(self,root,dataidxs=None,train=True,transform=None, target_transform=None, download=False, noise_type=None, noise_rate=0.1):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.data, self.target = self.Construct_Participant_Dataset()

    def Construct_Participant_Dataset(self):
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download, self.noise_type, self.noise_rate)
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_noisy_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            if self.train:
                data = cifar_dataobj.train_data
                target = np.array(cifar_dataobj.train_noisy_labels)
            else:
                data = cifar_dataobj.test_data
                target = np.array(cifar_dataobj.test_labels)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target
    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.data)

class Cifar100FL(Dataset):
    def __init__(self,root,dataidxs=None,train=True,transform=None, target_transform=None, download=False, noise_type=None, noise_rate=0):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.data, self.target = self.Construct_Participant_Dataset()
    def Construct_Participant_Dataset(self):
        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download, self.noise_type, self.noise_rate)
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            if self.train:
                data = cifar_dataobj.train_data
                target = np.array(cifar_dataobj.train_noisy_labels)
            else:
                data = cifar_dataobj.test_data
                target = np.array(cifar_dataobj.test_labels)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target
    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.data)


if __name__=='__main__':
    dataset_name = 'cifar_10'
    dataset_root = r'./'+dataset_name
    noise_type = None #[pairflip, symmetric]
    noise_rate = 0

    Cifar10FLParticipants = Cifar10FL(root=dataset_root,noise_type=noise_type, noise_rate=noise_rate, download=True)
    Cifar10FLTest  = Cifar10FL(root=dataset_root,train=False)

    # """补充"""
    #
    #
    # def unpickle(file):
    #     import pickle
    #     with open(file, 'rb') as fo:
    #         dict = pickle.load(fo, encoding='bytes')
    #     return dict
    #
    #
    # """自己补充检查label noise"""
    # data = Cifar10FLParticipants.data
    # label = Cifar10FLParticipants.target
    # for j in range(50000):
    #     imgdata = data[j, :, :, :]
    #     plt.cla()
    #     plt.imshow(np.transpose(imgdata, (0, 1, 2)))
    #     plt.pause(0.1)
    #     label_name = ['airplane', 'automobile', 'brid', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    #     print(label_name[label[j]])

    dataset_name = 'cifar_100'
    dataset_root = r'./'+dataset_name
    Cifar10FLPublic = Cifar100FL(root=dataset_root,noise_type=noise_type, noise_rate=noise_rate)
    print(len(Cifar10FLPublic))
    Cifar10FLTest = Cifar100FL(root=dataset_root,train=False)
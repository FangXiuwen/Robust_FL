import os
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

class Cifar_C(Dataset):
    def __init__(self, path, dataidxs=None, c_type=None, transform=None, corrupt_rate=0):
        self.dataidxs = dataidxs
        if c_type == 'random_noise':
            images = np.load(os.path.join(path, "aughfl_{}_{}.npy".format(c_type, corrupt_rate)))
        labels = np.load(os.path.join(path, "aughfl_labels.npy"))
        if self.dataidxs is not None:
            images = images[self.dataidxs]
            labels = labels[self.dataidxs]
        # self.images = concat_images[1:, ...].astype(np.uint8)
        self.images = images.astype(np.uint8)
        # self.labels = concat_labels[1:, ...]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img = self.images[index, ...]
        label = self.labels[index, ...]
        if self.transform:
            img = self.transform(img)
        return (img, label)

class CelebA(Dataset):
    def __init__(self,root,dataidxs=None,train=True,transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.Construct_Participant_Dataset()
    def Construct_Participant_Dataset(self):
        if self.train:
            celeba_dataobj = datasets.ImageFolder(self.root+'/img',transform=self.transform, target_transform=self.target_transform)
        else:
            celeba_dataobj = datasets.ImageFolder(self.root+'/img',transform=self.transform, target_transform=self.target_transform)
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = celeba_dataobj.train_data, np.array(celeba_dataobj.train_labels)
            else:
                data, target = celeba_dataobj.test_data, np.array(celeba_dataobj.test_labels)
        else:
            data = celeba_dataobj.imgs
            target = np.array(celeba_dataobj.targets)
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

class ImageFolder_custom(Dataset):
    def __init__(self,root,dataidxs=None,train=True,transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.Construct_Participant_Dataset()
    def Construct_Participant_Dataset(self):
        if self.train:
            tiny_imagenet_dataobj = datasets.ImageFolder(self.root+'/train',transform=self.transform, target_transform=self.target_transform)
        else:
            tiny_imagenet_dataobj = datasets.ImageFolder(self.root+'/val',transform=self.transform, target_transform=self.target_transform)
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = tiny_imagenet_dataobj.train_data, np.array(tiny_imagenet_dataobj.train_labels)
            else:
                data, target = tiny_imagenet_dataobj.test_data, np.array(tiny_imagenet_dataobj.test_labels)
        else:
            data = tiny_imagenet_dataobj.data
            target = np.array(tiny_imagenet_dataobj.targets)
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

    Cifar10FLParticipants = Cifar10FL(root=dataset_root,noise_type=noise_type, noise_rate=noise_rate)
    Cifar10FLTest  = Cifar10FL(root=dataset_root,train=False)
    dataset_name = 'cifar_100'
    dataset_root = r'./'+dataset_name
    Cifar10FLPublic = Cifar100FL(root=dataset_root,noise_type=noise_type, noise_rate=noise_rate)
    print(len(Cifar10FLPublic))
    Cifar10FLTest = Cifar100FL(root=dataset_root,train=False)
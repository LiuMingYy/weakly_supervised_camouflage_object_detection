#!/usr/bin/python3
#coding=utf-8

import os
import os.path as osp
import cv2
import torch
import numpy as np
try:
    from . import transform
except:
    import transform

from torch.utils.data import Dataset, DataLoader
from lib.data_prefetcher import DataPrefetcher

class Config(object):
    def __init__(self, **kwargs):
        if kwargs.get('label_dir') is None:
            kwargs['label_dir'] = 'Scribble'
        self.kwargs    = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

        if 'ECSSD' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.15, 112.48, 92.86]]])
            self.std       = np.array([[[ 56.36,  53.82, 54.23]]])
        elif 'DUTS' in self.kwargs['datapath']:
            self.mean      = np.array([[[124.55, 118.90, 102.94]]])
            self.std       = np.array([[[ 56.77,  55.97,  57.50]]])
        elif 'DUT-OMRON' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.61, 121.86, 114.92]]])
            self.std       = np.array([[[ 58.10,  57.16,  61.09]]])
        elif 'MSRA-10K' in self.kwargs['datapath']:
            self.mean      = np.array([[[115.57, 110.48, 100.00]]])
            self.std       = np.array([[[ 57.55,  54.89,  55.30]]])
        elif 'MSRA-B' in self.kwargs['datapath']:
            self.mean      = np.array([[[114.87, 110.47,  95.76]]])
            self.std       = np.array([[[ 58.12,  55.30,  55.82]]])
        elif 'SED2' in self.kwargs['datapath']:
            self.mean      = np.array([[[126.34, 133.87, 133.72]]])
            self.std       = np.array([[[ 45.88,  45.59,  48.13]]])
        elif 'PASCAL-S' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.02, 112.75, 102.48]]])
            self.std       = np.array([[[ 59.81,  58.96,  60.44]]])
        elif 'HKU-IS' in self.kwargs['datapath']:
            self.mean      = np.array([[[123.58, 121.69, 104.22]]])
            self.std       = np.array([[[ 55.40,  53.55,  55.19]]])
        elif 'SOD' in self.kwargs['datapath']:
            self.mean      = np.array([[[109.91, 112.13,  93.90]]])
            self.std       = np.array([[[ 53.29,  50.45,  48.06]]])
        elif 'THUR15K' in self.kwargs['datapath']:
            self.mean      = np.array([[[122.60, 120.28, 104.46]]])
            self.std       = np.array([[[ 55.99,  55.39,  56.97]]])
        elif 'SOC' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.48, 111.78, 101.27]]])
            self.std       = np.array([[[ 58.51,  56.73,  56.38]]])
        else:
            #raise ValueError
            self.mean = np.array([[[0.485*256, 0.456*256, 0.406*256]]])
            self.std = np.array([[[0.229*256, 0.224*256, 0.225*256]]])
            # self.std, self.mean = np.array([0.1861761914527739, 0.19748777412623036, 0.2032849354904543])[None,None]*255, np.array([0.3320486163733052, 0.432231354815684, 0.449829585669272])[None,None]*255

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_name = cfg.datapath.split('/')[-1]
        # 打开txt
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as name:
            lines_name = name.readlines()
        # 打开bad data txt
        with open('/mnt/disk/lym/COD10K/TrainDataset/myexp/bad_data_name.txt', 'r') as bad_name:
            lines_bad_name = bad_name.readlines()
        self.samples = []
        if cfg.mode == 'train':
            for line in lines_name:
                if line in lines_bad_name:
                    continue
                imagepath = cfg.datapath + '/' + 'TrainDataset' + '/Imgs/' + line.strip() + '.jpg'
                maskpath = cfg.datapath + '/' + 'TrainDataset' + f'/{cfg.label_dir}/' + line.strip() + '.png'
                syn_imagepath = cfg.datapath + '/' + 'TrainDataset' + '/myexp/' + 'synImgs/' + line.strip() + '.jpg'
                syn_maskpath = cfg.datapath + '/' + 'TrainDataset' + '/myexp/' + 'synScribble/' + line.strip() + '.png'
                regionpath = cfg.datapath + '/' + 'TrainDataset' + '/myexp/' + 'Simulated_Concave_Region/' + line.strip() + '.png'
                self.samples.append([imagepath, maskpath, syn_imagepath, syn_maskpath, regionpath])
        else:
            for line in lines_name:
                imagepath = cfg.datapath + '/Imgs/' + line.strip() + '.jpg'
                maskpath = cfg.datapath + '/GT/' + line.strip() + '.png'
                self.samples.append([imagepath, maskpath])

        if cfg.mode == 'train':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(320, 320),
                                                    transform.RandomHorizontalFlip(),
                                                    transform.RandomCrop(320, 320),
                                                    transform.ToTensor())

            self.syn_transform = transform.synCompose(transform.synNormalize(mean=cfg.mean, std=cfg.std),
                                                    transform.synResize(320, 320),
                                                    transform.synRandomHorizontalFlip(),
                                                    transform.synRandomCrop(320, 320),
                                                    transform.synToTensor())
        elif cfg.mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(320, 320),
                                                    transform.ToTensor()
                                                )
        else:
            raise ValueError

    def __getitem__(self, idx):
        if self.cfg.mode == 'train':
            imagepath, maskpath, syn_imagepath, syn_maskpath, regionpath = self.samples[idx]
            image = cv2.imread(imagepath).astype(np.float32)[:,:,::-1]
            mask = cv2.imread(maskpath).astype(np.float32)[:,:,::-1]
            syn_image = cv2.imread(syn_imagepath).astype(np.float32)[:,:,::-1]
            syn_mask = cv2.imread(syn_maskpath).astype(np.float32)[:,:,::-1]
            region = cv2.imread(regionpath).astype(np.float32)[:,:,::-1]

        else:
            imagepath, maskpath = self.samples[idx]
            image = cv2.imread(imagepath).astype(np.float32)[:,:,::-1]
            mask = cv2.imread(maskpath).astype(np.float32)[:,:,::-1]

        # unique_values = np.unique(region)
        # is_binary = len(unique_values) == 2 and unique_values[0] == 0 and unique_values[1] == 255
        # print('Is binary image:', is_binary)

        H, W, C             = mask.shape
        if self.cfg.mode == 'train':
            image, mask = self.transform(image, mask)
            syn_image, syn_mask, region = self.syn_transform(syn_image, syn_mask, region)
            mask[mask == 0.] = 255.
            mask[mask == 2.] = 0.
            syn_mask[syn_mask == 0.] = 255.
            syn_mask[syn_mask == 2.] = 0.
            return image, mask, syn_image, syn_mask, region, (H, W), maskpath.split('/')[-1]

            # unique_values = np.unique(region)
            # is_binary = len(unique_values) == 2
            # print('Is binary image:', is_binary)

        else:
            image, _         = self.transform(image, mask)
            mask = torch.from_numpy(mask.copy()).permute(2,0,1)
            mask = mask.mean(dim=0, keepdim=True)
            mask /= 255
            return image, mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='test', datapath='/mnt/disk/lym/COD10K/TestDataset/COD10K')
    data = Data(cfg)
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
    prefetcher = DataPrefetcher(loader, cfg)
    batch_idx = -1
    image, mask = prefetcher.next()
    image = image[0].permute(1,2,0).cpu().numpy()*cfg.std + cfg.mean
    mask  = mask[0].cpu().numpy().squeeze()
    plt.subplot(121)
    plt.imshow(np.uint8(image))
    plt.subplot(122)
    plt.imshow(mask)
    input()


#!/usr/bin/python3
#coding=utf-8

from configparser import Interpolation
import cv2
from torchvision.transforms.functional import rotate, InterpolationMode
import torch
import numpy as np

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, mask):
        for op in self.ops:
            image, mask = op(image, mask)
        return image, mask

class synCompose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, syn_image, syn_mask, region):
        for op in self.ops:
            syn_image, syn_mask, region = op(syn_image, syn_mask, region)
        return syn_image, syn_mask, region

class RGBDCompose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, depth, mask):
        for op in self.ops:
            image, depth, mask = op(image, depth, mask)
        return image, depth, mask


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, mask):
        image = (image - self.mean)/self.std
        # mask /= 255
        return image, mask

class RGBDNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, depth, mask):
        image = (image - self.mean)/self.std
        depth = (depth - self.mean)/self.std
        mask /= 255
        return image, mask

class synNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, syn_image, syn_mask, region):
        syn_image = (syn_image - self.mean)/self.std
        return syn_image, syn_mask, region

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

class synResize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, syn_image, syn_mask, region):
        # unique_values = np.unique(region)
        # is_binary = len(unique_values) == 2
        # print('Is binary image:', is_binary)

        syn_image = cv2.resize(syn_image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        syn_mask  = cv2.resize( syn_mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        region = cv2.resize(region, dsize=(self.W, self.H), interpolation=cv2.INTER_NEAREST)

        # unique_values = np.unique(region)
        # is_binary = len(unique_values) == 2
        # print('Is binary image:', is_binary)

        return syn_image, syn_mask, region

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        H,W,_ = image.shape
        xmin  = np.random.randint(W-self.W+1)
        ymin  = np.random.randint(H-self.H+1)
        image = image[ymin:ymin+self.H, xmin:xmin+self.W, :]
        mask  = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        return image, mask

class synRandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, syn_image, syn_mask, region):
        H,W,_ = syn_image.shape
        xmin  = np.random.randint(W-self.W+1)
        ymin  = np.random.randint(H-self.H+1)
        syn_image = syn_image[ymin:ymin+self.H, xmin:xmin+self.W, :]
        syn_mask  = syn_mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        region = region[ymin:ymin+self.H, xmin:xmin+self.W, :]
        return syn_image, syn_mask, region

class RandomHorizontalFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==1:
            image = image[:,::-1,:].copy()
            mask  =  mask[:,::-1,:].copy()
        return image, mask

class synRandomHorizontalFlip(object):
    def __call__(self, syn_image, syn_mask, region):
        if np.random.randint(2)==1:
            syn_image = syn_image[:,::-1,:].copy()
            syn_mask  =  syn_mask[:,::-1,:].copy()
            region    =    region[:,::-1,:].copy()
        return syn_image, syn_mask, region

class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        mask  = mask.permute(2, 0, 1)
        return image, mask.mean(dim=0, keepdim=True)

class synToTensor(object):
    def __call__(self, syn_image, syn_mask, region):
        syn_image = torch.from_numpy(syn_image)
        syn_image = syn_image.permute(2, 0, 1)
        syn_mask  = torch.from_numpy(syn_mask)
        syn_mask  = syn_mask.permute(2, 0, 1)
        region    = torch.from_numpy(region)
        region = region.permute(2, 0, 1)
        return syn_image, syn_mask.mean(dim=0, keepdim=True), region.mean(dim=0, keepdim=True)

class Flip:
    def __init__(self, flip_num):
        assert flip_num in [0,1,2]
        self.flip = flip_num
    def __call__(self, img, msk = None):
        if self.flip==1:
            img = img.flip(-2)
        elif self.flip==2:
            img = img.flip(-1)
        return img, msk

class Rotate:
    def __init__(self, rot_degree):
        self.rot = rot_degree
    def __call__(self, img, msk = None):
        img = rotate(img, self.rot, interpolation=InterpolationMode.BILINEAR)
        return img, msk

class RandomNoise:
    def __init__(self, noise_level):
        self.noise_level = noise_level
    def __call__(self, img, msk = None):
        noise = torch.randn_like(img) * self.noise_level
        img = img + noise
        return img, msk




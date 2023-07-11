import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from net import Net
from utils.dataloader import My_test_dataset
from data import dataset

root = '/mnt/disk/lym/COD10K'
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=320, help='testing size default 352')
# parser.add_argument('--pth_path', type=str, default='/mnt/disk/lym/pth/SARNet/ckpt/SARNet_v2/Net_epoch_best.pth')
parser.add_argument('--pth_path', type=str, default='/mnt/disk/lym/COD10K/result/COD10K/CRNet/model-best.pth')
opt = parser.parse_args()

cfg = [.15, 60, 16, 1]
w_ft, ft_st, topk,w_ftp = cfg
EXP_NAME = f'CRNet'
total_epoch = 150
cfg = dataset.Config(datapath=f'{root}', savepath=f'/mnt/disk/lym/COD10K/result/COD10K/{EXP_NAME}/', mode='train', batch=16, lr=1e-3, momen=0.9, decay=5e-4, epoch=total_epoch, label_dir = 'Scribble')
model = Net(cfg)
model.train(False)


# for _data_name in ['CAMO', 'COD10K', 'CHAMELEON',NC4K]:
for _data_name in ['COD10K']:
    # data_path = '/youtu_action_data/xiaobinhu/dataset_hitnet_cod/TestDataset/{}/'.format(_data_name)
    # save_path = '/youtu_action_data/xiaobinhu/dataset_hitnet_cod/res/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)

    # data_path = '/mnt/disk/lym/COD10K/TestDataset/COD10K'
    data_path = '/mnt/disk/lym/COD10K/TrainDataset'

    save_path = '/mnt/disk/lym/COD10K/result/COD10K/CRNet/exp1_train/'

    checkpoint = torch.load(opt.pth_path)
    model.load_state_dict(checkpoint)
    torch.cuda.set_device(0)
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    print('root',image_root,gt_root)
    test_loader = My_test_dataset(image_root, gt_root, opt.testsize)
    print('****',test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        print('***name',name)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        out, _, _, _, _, _ = model(image)
        res = F.upsample(out, size=gt.shape, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        # misc.imsave(save_path+name, res)
        # If `mics` not works in your environment, please comment it and then use CV2
        cv2.imwrite(save_path+name,res*255)

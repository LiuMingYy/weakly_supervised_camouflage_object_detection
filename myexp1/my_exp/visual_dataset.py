from skimage import io
import numpy as np
import cv2

def imadjust(x,a,b,c,d,gamma=1.5):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def enhance_gray(f):
    f = io.imread(f) # #依次读取rgb图片
    # f = (f*255.0).astype('uint8')
    res = imadjust(f,f.min(),f.max(),0,1)
    return res


data_dir = "/mnt/disk/lym/COD10K/TrainDataset/Scribble"
path_str = data_dir + '/*.png'
all_pic = io.ImageCollection(path_str,load_func=enhance_gray)

for i in range(len(all_pic)):  # 循环保存图片
    img = all_pic[i]
    name = all_pic.files[i].split('/')[-1]
    cv2.imwrite('/mnt/disk/lym/COD10K/TrainDataset/output/'+name,img*255)
    print(i)
    # io.imsave('/mnt/disk/lym/COD10K/TrainDataset/output/'+name.split('.')[0]+'jpg',all_pic[i])
    # io.imsave('./output/{}.png'.format(name),all_pic[i])

# cv.imwrite('/home/yz/catkin_camera/src/orbbec-ros-sdk/results/test/depth/{}.png'.format(name), (depth/32767)*255)

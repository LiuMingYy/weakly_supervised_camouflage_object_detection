import numpy as np
from PIL import Image

# 加载大的RGB图像
img = np.array(Image.open('C:/Users/ali/Desktop/fsdownload/新建文件夹/test.png'))

# 加载小的RGB纹理图像
texture = np.array(Image.open('C:/Users/ali/Desktop/fsdownload/新建文件夹/texture.png'))

# 定义不规则区域的布尔掩码
mask = np.zeros_like(img)
mask[100:200, 300:400, :] = 1

# 将纹理图像填充到不规则区域中
img = np.where(mask.astype(bool), texture, img)

# 保存修改后的RGB图像
Image.fromarray(img).save('C:/Users/ali/Desktop/fsdownload/result.png')
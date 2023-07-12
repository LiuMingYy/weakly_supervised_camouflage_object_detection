import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import io
import cv2
from scipy.spatial import distance, KDTree
from scipy.signal import convolve2d
import random
import math
import os


global bad_flag
bad_path=[]


def imadjust(x,a,b,c,d,gamma=1.5):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def enhance_gray(f):
    # f = io.imread(f)  # 依次读取rgb图片
    # f = (f*255.0).astype('uint8')
    res = imadjust(f,f.min(),f.max(),0,1)
    return res


def compute_distance(mode='Euclid', b=[], f=[]):
    (b_indice, f_indice) = (-1, -1)
    if mode == 'Euclid':
        # 计算距离
        dist_matrix = distance.cdist(b, f)  # o(n^n)
        # 获取距离最小的两个元素的索引
        min_index = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
        (b_indice, f_indice) = min_index

    if mode == 'KDtree':
        b_kdtree = KDTree(b)
        min_dist = np.inf
        min_dist_indices = None
        for i, fg_elem in enumerate(f):
            dist, idx = b_kdtree.query(fg_elem)
            if dist < min_dist:
                min_dist = dist
                min_dist_indices = (i, idx)
        (f_indice, b_indice) = min_dist_indices

    return b_indice, f_indice


def calculate_local_variance(coords, img, window_size):
    # 转换为Lab颜色空间
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    variances = []
    for coord in coords:
        # 提取局部窗口
        x, y = coord
        half_window = int(window_size / 2)
        window = lab_img[max(y - half_window, 0):min(y + half_window + 1, img.shape[0]),
                 max(x - half_window, 0):min(x + half_window + 1, img.shape[1]), :]
        # 计算方差
        variance = np.var(window, axis=(0, 1))
        variances.append(variance)
    return variances


def find_minimum_variance(coords, variances):
    min_variance = None
    min_coord = None
    for i, variance in enumerate(variances):
        if min_variance is None or np.sum(variance) < np.sum(min_variance):
            min_variance = variance
            min_coord = coords[i]
    return min_coord


def select_point(res, img):
    # 使用nonzero函数找出非零元素的索引
    global bad_flag
    nonzero_indices = np.nonzero(res)

    # 显示非零元素的索引
    # print("Non-zero indices: ", nonzero_indices)

    bg = []
    fg = []

    # 遍历非零元素并显示它们的值和位置
    for i in range(len(nonzero_indices[0])):
        row = nonzero_indices[0][i]
        col = nonzero_indices[1][i]
        value = res[row, col]
        if value[0] == 2:
            bg.append((row, col))
        else:
            fg.append((row, col))

    # 计算两个元组列表之间的距离矩阵
    if len(fg) == 0 or len(bg) == 0:
        start_point = (-1,-1)
        end_point = (-1,-1)
        bad_flag = True
    else:
        # 计算最小方差
        # variances = calculate_local_variance(bg, img, window_size=90)
        # start_point = find_minimum_variance(bg, variances)
        #
        # end_point = random.choice(fg)

        # 计算距离
        bg_indice, fg_indice = compute_distance(mode='KDtree', b=bg, f=fg)
        start_point = bg[bg_indice]
        end_point = fg[fg_indice]

    return start_point, end_point


def texture_generation(img=None, size=15, x=0, y=0):
    # 纹理裁剪
    texture_size = (size, size)

    left, top = x - texture_size[0] // 2, y - texture_size[1] // 2
    right, bottom = left + texture_size[0], top + texture_size[1]
    texture = img.crop((left, top, right, bottom))

    # 上
    up_mirror = texture.transpose(Image.FLIP_TOP_BOTTOM)
    # 下
    down_mirror = texture.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_180)
    # 左
    left_mirror = texture.transpose(Image.FLIP_LEFT_RIGHT)
    # 右
    right_mirror = texture.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180)
    # 左上
    up_left_mirror = up_mirror.transpose(Image.FLIP_LEFT_RIGHT)
    # 右上
    up_right_mirror = up_mirror.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180)
    # 左下
    down_left_mirror = down_mirror.transpose(Image.FLIP_LEFT_RIGHT)
    # 右下
    down_right_mirror = down_mirror.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180)

    # 拼接
    new_w = 3 * texture.width
    new_h = 3 * texture.height

    new_texture = Image.new('RGB', (new_w, new_h))

    new_texture.paste(up_left_mirror, (0, 0))  # 左上
    new_texture.paste(up_mirror, (texture.width, 0))  # 上
    new_texture.paste(up_right_mirror, (2 * texture.width, 0))  # 右上
    new_texture.paste(left_mirror, (0, texture.height))  # 左
    new_texture.paste(texture, (texture.width, texture.height))  # 中
    new_texture.paste(right_mirror, (2 * texture.width, texture.height))  # 右
    new_texture.paste(down_left_mirror, (0, 2 * texture.height))  # 左下
    new_texture.paste(down_mirror, (texture.width, 2 * texture.height))  # 下
    new_texture.paste(down_right_mirror, (2 * texture.width, 2 * texture.height))  # 右下

    arr = np.array(new_texture)
    return arr


def path_generation(num_points=100, start_point=(0,0), end_point=(0,0), img_arr=None, scribble_img_arr=None):

    # 由start_point至end_point的直线
    x = np.linspace(start_point[0], end_point[0], num_points, dtype=np.int32)
    y = np.linspace(start_point[1], end_point[1], num_points, dtype=np.int32)
    points = np.column_stack((y, x))

    # 将路径上的像素坐标记录到一个元组列表中
    path_list = [(point[0], point[1]) for point in points]
    height, width, channels = img_arr.shape
    blank_image = np.zeros((height, width, channels), dtype=np.uint8)
    N = len(path_list)

    for i in range(len(path_list)):

        # 中心坐标
        center_point = path_list[i]

        # 定义窗口大小
        k = random.randint(10, 15)
        beta_1 = random.randint(-1, 1)
        beta_2 = random.randint(1, 2)
        d = k * (1 + beta_1 * (i / N) * (math.sin(beta_2 * (i / N) * math.pi)))
        if random.random() < 0.5:
            d = math.floor(d)
        else:
            d = math.ceil(d)
        window_size = (d, d)

        # 计算窗口左上角和右下角坐标
        up_left_x = max(center_point[0] - window_size[0] // 2, 0)
        up_left_y = max(center_point[1] - window_size[1] // 2, 0)
        down_right_x = min(center_point[0] + window_size[0] // 2, width)
        down_right_y = min(center_point[1] + window_size[1] // 2, height)

        # 将窗口内像素赋值
        img_arr[up_left_y:down_right_y, up_left_x:down_right_x, :] = 255
        blank_image[up_left_y:down_right_y, up_left_x:down_right_x, :] = 255
        scribble_img_arr[up_left_y:down_right_y, up_left_x:down_right_x, :] = 2

    return path_list, img_arr, blank_image, scribble_img_arr


def img_mix_texture(img_arr=None,blank_image=None, texture_arr=None):
    # 根据掩膜图像裁剪不规则区域
    cropped_irregular_area = cv2.bitwise_and(img_arr, blank_image)
    # 将小纹理图像平铺到裁剪后的不规则区域图像
    irregular_area_height, irregular_area_width, _ = cropped_irregular_area.shape
    texture_height, texture_width, _ = texture_arr.shape
    repeat_x = int(np.ceil(irregular_area_width / texture_width))
    repeat_y = int(np.ceil(irregular_area_height / texture_height))
    tiled_texture = np.tile(texture_arr, (repeat_y, repeat_x, 1))
    tiled_texture = tiled_texture[:irregular_area_height, :irregular_area_width]
    # 将平铺好的小纹理图像与裁剪后的不规则区域图像进行按位与操作
    filled_irregular_area = cv2.bitwise_and(cropped_irregular_area, tiled_texture)
    # 将填充好纹理的不规则区域图像与原始大图像合并
    large_image_with_filled_area = cv2.bitwise_and(img_arr, cv2.bitwise_not(blank_image))
    result_image = cv2.bitwise_or(large_image_with_filled_area, filled_irregular_area)
    # 应用随机羽化操作来更好地合并纹理和原始图像
    mask = blank_image
    blur_kernel_size = 15
    blur_sigma = 30
    cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), blur_sigma, dst=mask)
    alpha = mask.astype(float) / 255
    img = cv2.convertScaleAbs(result_image * (1 - alpha) + result_image * alpha)

    return img


def main_method(image_path='', scribble_path='', save_path=''):

    # 加载RGB图像 and Scribble图像
    image = Image.open(image_path)
    img_arr = np.array(image)
    scribble_image = Image.open(scribble_path)
    scribble_img_arr = np.array(scribble_image)

    # 端点选择
    start_point, end_point = select_point(scribble_img_arr, img_arr)
    if start_point == (-1,-1) or end_point == (-1,-1):
        bad_path.append(image_path.split('/')[-1].split('.')[0])

    # 路径生成
    if not bad_flag:
        path_list, img_arr, blank_image, scribble_img_arr = path_generation(start_point=start_point,
                                                                                 end_point=end_point,
                                                                                 img_arr=img_arr,
                                                                                 scribble_img_arr=scribble_img_arr)

        # 纹理生成
        (start_x, start_y) = path_list[0]
        texture_arr = texture_generation(image, size=15, x=start_x, y=start_y)

        # 合成
        mask = np.copy(blank_image)
        result_image = img_mix_texture(img_arr, mask, texture_arr)

        # 保存
        name = image_path.split('/')[-1].split('.')[0]
        # 合成rgb图像
        rgb_path = save_path + 'synImgs'
        if not os.path.exists(rgb_path):  # 检查目录是否存在
            os.makedirs(rgb_path)  # 创建目录
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(rgb_path + '/' + name + '.jpg', result_image)
        # 合成Scribble图像
        png_path = save_path + 'synScribble'
        if not os.path.exists(png_path):  # 检查目录是否存在
            os.makedirs(png_path)  # 创建目录
        cv2.imwrite(png_path + '/' + name + '.png', scribble_img_arr)
        # 可视化合成Scribble图像
        vis_path = save_path + 'synScribble_vis'
        if not os.path.exists(vis_path):  # 检查目录是否存在
            os.makedirs(vis_path)  # 创建目录
        cv2.imwrite(vis_path + '/' + name + '.png', enhance_gray(scribble_img_arr) * 255)
        # Simulated Concave Region
        SCR_path = save_path + 'Simulated_Concave_Region'
        if not os.path.exists(SCR_path):  # 检查目录是否存在
            os.makedirs(SCR_path)  # 创建目录
        cv2.imwrite(SCR_path + '/' + name + '.png', blank_image)


if __name__ == '__main__':
    savePath = 'C:/Users/ali/Desktop/fsdownload/myexp/'
    trainPath = 'C:/Users/ali/Desktop/fsdownload/train'
    scribblePath = 'C:/Users/ali/Desktop/fsdownload/Scribble'
    i=0
    for filename in os.listdir(trainPath):
        bad_flag = False
        length = len(os.listdir(trainPath))
        print("{} : {}".format(i, length))
        i += 1
        imagePath = trainPath + '/' + filename
        scribble_path = scribblePath + '/' + filename.split('.')[0]+'.png'
        main_method(image_path=imagePath, scribble_path=scribble_path, save_path=savePath)
        if bad_flag:
            print("bad data!!!")
    print("num of bad: ", len(bad_path))
    txt_path = 'C:/Users/ali/Desktop/fsdownload/myexp/bad_data_name.txt'
    if not os.path.exists(os.path.dirname(txt_path)):
        os.makedirs(os.path.dirname(txt_path))

    with open(txt_path, 'w') as file:
        for item in bad_path:
            file.write(item + '\n')





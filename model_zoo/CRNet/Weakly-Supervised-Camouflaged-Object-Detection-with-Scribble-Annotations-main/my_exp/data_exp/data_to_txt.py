import os

# 指定要遍历的文件夹路径
folder_path = '/mnt/disk/lym/COD10K/TestDataset/COD10K/Imgs'

# 指定要输出的txt文件路径
txt_file_path = '/mnt/disk/lym/COD10K/TestDataset/COD10K/test.txt'

# 遍历指定文件夹及其子文件夹中的所有图片
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
image_names = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if any(file.endswith(extension) for extension in image_extensions):
            image_names.append(file.split('.')[0])

# 将图片名称按列写入指定的txt文件中
with open(txt_file_path, 'w') as f:
    for name in image_names:
        f.write(name + '\n')
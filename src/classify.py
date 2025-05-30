#这个是拿来做图片分类的，具体看代码

import os
import shutil



# 创建数据集目录结构
original_dataset_dir = r'Project1/train'
base_dir = r'Project1/data/train'
os.makedirs(base_dir, exist_ok=True)

# 创建训练、验证、测试集目录
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

for dir_path in [train_dir, validation_dir, test_dir]:
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'cats'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'dogs'), exist_ok=True)

# 复制图像到对应目录
def copy_images(src_pattern, start_idx, end_idx, dst_dir):
    fnames = [f'{src_pattern}{i}.jpg' for i in range(start_idx, end_idx)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(dst_dir, fname)
        shutil.copyfile(src, dst)

# 复制猫的图像
copy_images('cat.', 0, 1000, os.path.join(train_dir, 'cats'))
copy_images('cat.', 1000, 1500, os.path.join(validation_dir, 'cats'))
copy_images('cat.', 1500, 2000, os.path.join(test_dir, 'cats'))

# 复制狗的图像
copy_images('dog.', 0, 1000, os.path.join(train_dir, 'dogs'))
copy_images('dog.', 1000, 1500, os.path.join(validation_dir, 'dogs'))
copy_images('dog.', 1500, 2000, os.path.join(test_dir, 'dogs'))

# 验证数据集划分
print('训练集猫图像数量:', len(os.listdir(os.path.join(train_dir, 'cats'))))
print('训练集狗图像数量:', len(os.listdir(os.path.join(train_dir, 'dogs'))))
print('验证集猫图像数量:', len(os.listdir(os.path.join(validation_dir, 'cats'))))
print('验证集狗图像数量:', len(os.listdir(os.path.join(validation_dir, 'dogs'))))
print('测试集猫图像数量:', len(os.listdir(os.path.join(test_dir, 'cats'))))
print('测试集狗图像数量:', len(os.listdir(os.path.join(test_dir, 'dogs'))))
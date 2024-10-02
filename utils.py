import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import random
import torch

def count_acc(logits, label, n_way):
    '''count Acc(normalized)'''
    pred = torch.argmax(logits, dim=1)
    Acc_class = ((pred == label).sum().item()) / label.size(0) # 获取预测正确的样本数
    Acc_rc = 1 / n_way
    correct = (Acc_class - Acc_rc) / (1 - Acc_rc)
    return correct  # 计算精确度


# 根据传入的数据地址列表，拿到所有文件的相对路径
def generate_file_list(list_dir):
    extension = ('.png', '.PNG', 'JPG', 'JPEG')#用于检查文件名是否以指定的扩展名结尾
    file_list = []
    for path in list_dir:
        for filename in os.listdir(path):
            if filename.endswith(extension):
                fullPath = os.path.join(path, filename)
                file_list.append(fullPath)
    return file_list

def adjust_learning_rate(epoch, optimizer, lr_decay_epochs = [60,80], learning_rate = 0.05, lr_decay_rate = 0.1):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
            
def get_transforms(mode = 'embedding'):
    # 使用列表生成式计算归一化参数

    # 创建组合转换
    if mode == 'meta':
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize 应该接受整数或者(h, w)的元组
            transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
                                 std=[70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0])
        ])
    elif 'embedding' == mode:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(128, padding=8),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
                                 std=[70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0])
        ])
    return transform

def files_to_tensor(img_files, mode = 'embedding'):
    imgs = []
    for file in img_files:
        img = Image.open(file)
        img = img.convert(mode="RGB")
        transform = get_transforms(mode = mode) # embedding operate
        # transform = get_transforms(mode='train')  #meta-training
        img = transform(img)
        imgs.append(img)
    return torch.stack(imgs, dim=0)



def cls_files(catgory, root):
    cls_to_files = {}
    for cat in catgory: #bottle、cable、....
        for cls in os.listdir(os.path.join(root, cat)) :
            if cls == 'train':
                file_dir = os.path.join(root, cat, cls, 'good')
                if f'{cat}_good' not in cls_to_files.keys():
                    cls_to_files[f'{cat}_good'] = generate_file_list([file_dir])
                else:
                    cls_to_files[f'{cat}_good'] += generate_file_list([file_dir])
            elif cls == 'test':
                for way in os.listdir(os.path.join(root, cat, cls)):
                    if way == 'good':
                        file_dir = os.path.join(root, cat, cls, way)
                        if f'{cat}_good' not in cls_to_files.keys():
                            cls_to_files[f'{cat}_good'] = generate_file_list([file_dir])
                        else:
                            cls_to_files[f'{cat}_good'] += generate_file_list([file_dir])
                    else:
                        file_dir = os.path.join(root, cat, cls, way)
                        cls_to_files[f'{cat}_{way}'] = generate_file_list([file_dir])
    return cls_to_files

def cls2files(catgory, root):
    cls_to_files = {}
    for cat in catgory: #bottle、cable、....
        for cls in os.listdir(os.path.join(root, cat)) :
            if cls == 'test':
                for way in os.listdir(os.path.join(root, cat, cls)):
                    if way != 'good':
                        file_dir = os.path.join(root, cat, cls, way)
                        cls_to_files[f'{cat}_{way}'] = generate_file_list([file_dir])
    return cls_to_files


def get_teacher_name(model_path):
    """parse to get teacher model name"""
    segment = model_path.split('/')[-1].split('_')[0]
    return segment


def create_model(name, n_cls = 50):
    model = model_dict[name](num_classes=n_cls)
    return model

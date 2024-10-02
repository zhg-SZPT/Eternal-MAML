import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils import files_to_tensor, generate_file_list
from category_split import folder8


def cls2files(catgory, root):
    cls_to_files = {}
    for cat in catgory: #bottle、cable、....
        for cls in os.listdir(os.path.join(root, cat)) :
            if cls == 'test':
                for way in os.listdir(os.path.join(root, cat, cls)):
                    file_dir = os.path.join(root, cat, cls, way)
                    cls_to_files[f'{cat}_{way}'] = generate_file_list([file_dir])
    return cls_to_files

class MvTec(Dataset):
    def __init__(self, root = '../data/mvtec', mode = 'train'):
        # 设置随机种子
        # random.seed(2)
        self.root = root
        self.train_cat = folder8['meta-train']
        self.cls_to_files = cls2files(self.train_cat, self.root)
        self.target_to_files = {}
        self.list_to_files = []
        #
        for cls in self.cls_to_files.keys():
            # random.shuffle(self.cls_to_files[cls])
            if 'train' == mode:
                self.target_to_files[cls] = self.cls_to_files[cls][:int(len(self.cls_to_files[cls]) * 0.8)]
            elif 'val' == mode:
                self.target_to_files[cls] = self.cls_to_files[cls][int(len(self.cls_to_files[cls]) * 0.8):]

        for idx, (_, files) in enumerate(self.target_to_files.items()):
            for file in files:
                self.list_to_files.append([idx, file])
        random.shuffle(self.list_to_files)


    def __getitem__(self, idx):
        label, data = self.list_to_files[idx]
        # return torch.tensor([label],device='cuda'), files_to_tensor([data]).to('cuda')
        return torch.tensor([label],device='cuda'), files_to_tensor([data]).to('cuda')


    def __len__(self):
        # pass
        return len(self.list_to_files)


















































# from category_split import train_cls, val_cls, test_cls

# a = 1
# class MvTec(Dataset):
#     def __init__(self, root = 'D:\\datasets\\meta-learning\\mvtec'):
#         self.root = root
#         self.remove_cls = 'bottle'
#         self.category = [name for name in os.listdir(self.root)]
#         self.category.remove(self.remove_cls)
#
#         self.cls_to_files = cls_files(self.category, self.root)
#         self.target_to_files = {}
#         self.list_to_files = []
#
#         for cls in train_cls + val_cls + test_cls:
#             self.target_to_files[cls] = self.cls_to_files[cls]
#
#         for idx, (_, files) in enumerate(self.target_to_files.items()):
#             for file in files:
#                 self.list_to_files.append([idx, file])
#         random.shuffle(self.list_to_files)
#
#
#     def __getitem__(self, idx):
#         label, data = self.list_to_files[idx]
#         return torch.tensor([label],device='cuda'), files_to_tensor([data]).to('cuda')
#
#
#     def __len__(self):
#         # pass
#         return len(self.list_to_files)
#
#     # def test(self):
#     #     pass
#     #     # return self.train_good, self.test_good


# class MvTec_normal_abnormal(Dataset):
#     def __init__(self, root='D:\\datasets\\meta-learning\\mvtec'):
#
#         self.root = root
#         self.remove_cls = 'bottle'
#         self.category = [name for name in os.listdir(self.root)]
#         self.category.remove(self.remove_cls)
#         self.good = ['test\\good', 'train\\good']
#         self.cls_to_files = {}
#         self.list_to_files = []
#         for cat in self.category:
#             abnor_cls = [os.path.join(self.root, cat, 'test', c) for c in os.listdir(os.path.join(self.root, cat, 'test')) if c is not 'good']
#             nor_files = [os.path.join(self.root, cat, 'test', 'good', c) for c in os.listdir(os.path.join(self.root, cat, 'test', 'good'))]
#             nor_files += [os.path.join(self.root, cat, 'train', 'good',c) for c in os.listdir(os.path.join(self.root, cat, 'train', 'good'))]
#             random.shuffle(nor_files)
#             abnor_files = [os.path.join(self.root, cat, 'test', cls, file) for cls in abnor_cls for file in os.listdir(cls)]
#             nor_files = nor_files[:len(abnor_files)]
#             if 0 not in self.cls_to_files.keys():
#                 self.cls_to_files[0] = nor_files
#             else:
#                 self.cls_to_files[0] += nor_files
#             if 1 not in self.cls_to_files.keys():
#                 self.cls_to_files[1] = abnor_files
#             else:
#                 self.cls_to_files[1] += abnor_files
#
#         for idx, files in self.cls_to_files.items():
#             for file in files:
#                 self.list_to_files.append([idx, file])
#         random.shuffle(self.list_to_files)
#
#     def __getitem__(self, idx):
#         label, data = self.list_to_files[idx]
#         return torch.tensor([label],device='cuda'), files_to_tensor([data]).to('cuda')
#     def __len__(self):
#         return len(self.list_to_files)

# def cls2good(root):
#     category = [name for name in os.listdir(root)]
#     cat2good = {}
#     for cat in category:
#         cat2good[cat] = []
#         for file in os.listdir(os.path.join(root, cat, 'train', 'good')):
#             cat2good[cat].append(os.path.join(root, cat, 'train', 'good', file))
#     return cat2good



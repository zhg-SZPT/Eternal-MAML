import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils import files_to_tensor, cls_files
from category_split import folder1

# a = 1
class MvTec(Dataset):
    def __init__(self, root = '../data/mvtec', mode = 'train'):

        self.root = root
        self.train_cat = folder1['meta_test']
        self.cls_to_files = cls_files(os.listdir(self.root), self.root)
        self.target_to_files = {}
        self.list_to_files = []
        for cls in self.train_cat:
            self.target_to_files[cls] = self.cls_to_files[cls]

        for idx, (_, files) in enumerate(self.target_to_files.items()):
            if 'train' == mode:
                for file in files[:int(len(files) * 0.8)]:
                    self.list_to_files.append([idx, file])
            elif 'test' == mode:
                for file in files[int(len(files) * 0.8):]:
                    self.list_to_files.append([idx, file])
        random.shuffle(self.list_to_files)
        # random.shuffle(self.tlist_to_files)


    def __getitem__(self, idx):
        label, data = self.list_to_files[idx]
        return torch.tensor([label],device='cuda'), files_to_tensor([data]).to('cuda')


    def __len__(self):
        # pass
        return len(self.list_to_files)

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

def cls2good(root):
    category = [name for name in os.listdir(root)]
    cat2good = {}
    for cat in category:
        cat2good[cat] = []
        for file in os.listdir(os.path.join(root, cat, 'train', 'good')):
            cat2good[cat].append(os.path.join(root, cat, 'train', 'good', file))
    return cat2good
# class MvTec(Dataset):
#     def __init__(self, root = 'D:/datasets/meta-learning/mvtec', mode = 'train'):
#         self.root = root
#         self.category = [name for name in os.listdir(self.root)]
#         self.cat2good = cls2good(self.root)
#         self.list_to_files = []
#         for idx, (_, files) in enumerate(self.cat2good.items()):
#             if('train' == mode):
#                 for file in files[:int(len(files) * 0.6)]:
#                     self.list_to_files.append([idx, file])
#             elif('val' == mode):
#                 for file in files[int(len(files) * 0.6) : int(len(files) * 0.8)]:
#                     self.list_to_files.append([idx, file])
#             elif('test' == mode):
#                 for file in files[int(len(files) * 0.8): ]:
#                     self.list_to_files.append([idx, file])
#             else:
#                 raise "please input valid mode"
#
#         random.shuffle(self.list_to_files)
#
#
#     def __getitem__(self, idx):
#
#         label, data = self.list_to_files[idx]
#         return torch.tensor([label],device='cuda'), files_to_tensor([data]).to('cuda')
#
#
#     def __len__(self):
#         return len(self.list_to_files)

if __name__ == "__main__":
    train_data = MvTec(mode='train')
    test_data = MvTec(mode='test')
    # Dataset = MvTec_normal_abnormal()
    a = 1
    # loader = DataLoader(Dataset, batch_size=64)

    # val_Dataset = MvTec(args, mode='val')
    # # label, file = Dataset[0]
    # train_dataLoader = DataLoader(train_Dataset, batch_size=64)
    # for idx, (target, input) in enumerate(train_dataLoader):
    #     print(idx)
    #     a = 1
    # a = 1




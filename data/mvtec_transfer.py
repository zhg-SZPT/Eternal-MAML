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


def cls2good(root):
    category = [name for name in os.listdir(root)]
    cat2good = {}
    for cat in category:
        cat2good[cat] = []
        for file in os.listdir(os.path.join(root, cat, 'train', 'good')):
            cat2good[cat].append(os.path.join(root, cat, 'train', 'good', file))
    return cat2good


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




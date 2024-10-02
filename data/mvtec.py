import os
import random
import argparse
from utils import files_to_tensor, cls_files, cls2files
from torch.utils.data import Dataset, DataLoader

from category_split import folder2
def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    # setting for meta-mvtec data
    parser.add_argument('--data_root', type=str, default='../data/mvtec', help='path to data root')

    parser.add_argument('--n_way', type=int, default = 3, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--k_shots', type=int, default = 3, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--k_queries', type=int, default = 5, metavar='N',
                        help='Number of query in test')
    opt = parser.parse_args()
    return opt


class MvTec():
    """
    划分MVTec数据集, 集合了所有缺陷的种类
    """

    def __init__(self):
        self.opt = parse_option()
        self.root = self.opt.data_root
        self.n_way = self.opt.n_way
        self.k_shot = self.opt.k_shots  
        self.k_query = self.opt.k_queries
        self.train_cat = folder2['meta-train']
        self.train_2_files = cls2files(self.train_cat, self.root)

    def sample(self, mode='train', test_cat = None):
        data_dict = {}

        if 'train' == mode:
            random_cat = random.choice(self.train_cat)

            random_way = random.sample(cls2files([random_cat], self.root).keys(), self.n_way)
            label = [i for i in range(self.n_way)]
            random.shuffle(label)  # random label
            for l, way in zip(label, random_way):
                data_dict[l] = random.sample(self.train_2_files[way], self.k_shot + self.k_query)

        elif 'test' == mode:
            test_2_files = cls2files([test_cat], self.root)
            random_way = random.sample(test_2_files.keys(), self.n_way)
            label = [i for i in range(self.n_way)]
            random.shuffle(label)  # random label
            for l, way in zip(label, random_way):
                data_dict[l] = random.sample(test_2_files[way], self.k_shot + self.k_query)
        else:
            raise "input error!"

        support_list = [(i, file) for i, files in data_dict.items() for file in files[:self.k_shot]]
        query_list = [(i, file) for i, files in data_dict.items() for file in files[self.k_shot:]]
        random.shuffle(support_list), random.shuffle(query_list)

        support_label = []
        query_label = []
        support_data = []
        query_data = []
        for tup in support_list:
            label, data = tup
            support_data.append(data)
            support_label.append(label)

        for tup in query_list:
            label, data = tup
            query_data.append(data)
            query_label.append(label)

        return files_to_tensor(support_data, mode='meta'), support_label, files_to_tensor(query_data, mode='meta'), query_label


class MvTec_test(Dataset):
    def __init__(self, root = '../data/mvtec', mode = 'train'):
        # 设置随机种子
        # random.seed(2)
        self.root = root
        self.test_cat = folder7['meta-test']
        self.cls_to_files = cls2files(self.test_cat, self.root)
        self.target_to_files = {}
        self.list_to_files = []
        #
        for cls in self.cls_to_files.keys():
            if('train' == mode):
                self.target_to_files[cls] = self.cls_to_files[cls][ :int(len(self.cls_to_files[cls]) * 0.8)]
            elif('test' == mode):
                self.target_to_files[cls] = self.cls_to_files[cls][int(len(self.cls_to_files[cls])* 0.8): ]
        for idx, (_, files) in enumerate(self.target_to_files.items()):
            for file in files:
                self.list_to_files.append([idx, file])
        random.shuffle(self.list_to_files)


    def __getitem__(self, idx):
        label, data = self.list_to_files[idx]
        return torch.tensor([label],device='cuda'), files_to_tensor([data]).to('cuda')


    def __len__(self):
        # pass
        return len(self.list_to_files)
# class metaMvTec():

#     """
#     划分MVTec数据集, 集合了所有缺陷的种类
#     """

#     def __init__(self):
#         self.opt = parse_option()
#         self.root = self.opt.data_root
#         self.n_way = self.opt.n_way
#         self.k_shot = self.opt.k_shots
#         self.k_query = self.opt.k_queries
#         self.train_category = folder1['meta_train']
#         self.test_category = folder1['meta_test']
#         self.cls_to_files = cls_files(os.listdir(self.root), self.root)
#     def sample(self, mode='train'):
#         data_dict = {}
#         if 'train' == mode:
#             random_way = random.sample(self.train_category, self.n_way)
#             label = [i for i in range(self.n_way)]
#             random.shuffle(label)  # random label
#             for l, way in zip(label, random_way):
#                 data_dict[l] = random.sample(self.cls_to_files[way], self.k_shot + self.k_query)
#         elif 'test' == mode:
#             random_way = random.sample(self.test_category, self.n_way)
#             label = [i for i in range(self.n_way)]
#             random.shuffle(label)  # random label
#             for l, way in zip(label, random_way):
#                 data_dict[l] = random.sample(self.cls_to_files[way], self.k_shot + self.k_query)
#         else:
#             raise "input error!"


#         support_list = [(i, file) for i, files in data_dict.items() for file in files[:self.k_shot]]
#         query_list = [(i, file) for i, files in data_dict.items() for file in files[self.k_shot:]]
#         random.shuffle(support_list), random.shuffle(query_list)

#         support_label = []
#         query_label = []
#         support_data = []
#         query_data = []
#         for tup in support_list:
#             label, data = tup
#             support_data.append(data)
#             support_label.append(label)

#         for tup in query_list:
#             label, data = tup
#             query_data.append(data)
#             query_label.append(label)

#         return files_to_tensor(support_data, mode = 'meta'), support_label, files_to_tensor(query_data, mode = 'meta'), query_label
# #         return support_data, support_label, query_data, query_label
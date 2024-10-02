import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import random
from Data.mvtec import MvTec
from torch.utils.data import DataLoader
import torchvision
from collections import OrderedDict
from category_split import *
def count_acc(logits, label):
    '''count Acc(normalized)'''
    pred = torch.argmax(logits, dim=1)
    correct = ((pred == label).sum().item()) / label.size(0)  # 获取预测正确的样本数
    return correct  # 计算精确度


# 数据
# train_data = MvTec(mode='train')
# val_data = MvTec(mode='val')
# trainLoader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
# valLoader = DataLoader(val_data, batch_size=64, shuffle=True, drop_last=True)


test_train_data = MvTec_test(mode = 'train')
testtrainLoader = DataLoader(test_train_data, batch_size=64, shuffle=True, drop_last=True)
test_test_data = MvTec_test(mode = 'test')
testLoader = DataLoader(test_test_data, batch_size=64, shuffle=True, drop_last=True)


# 网络
n_way = 3
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, n_way)
model = model.to('cuda')

# 损失函数
criterion = torch.nn.CrossEntropyLoss().to('cuda')

# 优化器
optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 5e-2)

if __name__ == '__main__':
    epochs = 100
    target_acc = 0
    best_val_acc = 0
    best_params = OrderedDict()
    model.train()
    for epoch in range(1, epochs + 1):
        if(epoch == 1 or epoch % 10 == 0):
            random_industrail = random.sample(folder8['meta-test'], 3) #随机抽取三个类别
            
        print(f'--------epoch{epoch}--------')
        print('==> Training...')
        loss_count = 0
        acc_count = 0

        for target, input in tqdm(testtrainLoader):
            # ===================forward=====================
            input = input.reshape(-1, 3, 128, 128)
            target = target.reshape(-1, )
            output = model(input)
            loss = criterion(output, target)
            acc = count_acc(output, target)
            loss_count += loss.item()
            acc_count += acc
            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'train_loss = {loss_count / len(testtrainLoader):.2f}')
        print(f'train_accuracy = {acc_count / len(testtrainLoader):.2f}')



        if (epoch % 10 == 0):
            vacc_count = 0
            print('----------validation------------')
            for vtarget, vinput in tqdm(testLoader):
                # ===================forward=====================
                vinput = vinput.reshape(-1, 3, 128, 128)
                vtarget = vtarget.reshape(-1, )
                output = model(vinput)
                vacc = count_acc(output, vtarget)
                vacc_count += vacc
            if(best_val_acc < (vacc_count / len(testLoader))):
                best_val_acc = (vacc_count / len(testLoader))
                best_params = OrderedDict(model.named_parameters())
            print(f'val_acc = {vacc_count / len(testLoader):.2f}')

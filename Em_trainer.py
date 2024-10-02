import os
import torch.nn
from tqdm import tqdm
from Data.mvtec_embedding import MvTec
from torch.utils.data import DataLoader
from utils import adjust_learning_rate
from networks.resNet import seresnet12


def count_acc(logits, label):
    '''count Acc(normalized)'''
    pred = torch.argmax(logits, dim=1)
    correct = ((pred == label).sum().item()) / label.size(0)  # 获取预测正确的样本数
    return correct  # 计算精确度



#数据
train_data = MvTec(mode='train')
val_data = MvTec(mode='val')
trainLoader = DataLoader(train_data, batch_size = 64, shuffle=True, drop_last=True)
valLoader = DataLoader(val_data, batch_size = 64, shuffle=True, drop_last=True)

#网络
model = seresnet12(avg_pool = True, drop_rate = 0.1, dropblock_size = 2, num_classes = 42)
model = model.to('cuda')

#损失函数
criterion = torch.nn.CrossEntropyLoss().to('cuda')

#优化器
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, weight_decay = 5e-4, momentum = 0.9)

if __name__ == '__main__':
    epochs= 100
    target_acc = 0
    
    for epoch in range(1, epochs + 1):
        print(f'--------epoch{epoch}--------')
        print('==> Training...')
        model.train()
        adjust_learning_rate(epoch, optimizer)
        loss_count = 0
        acc_count = 0

        for target, input in tqdm(trainLoader):
            # ===================forward=====================
            input = input.reshape(-1, 3, 128, 128)
            target = target.reshape(-1,)
            output = model(input)
            loss = criterion(output, target)
            acc = count_acc(output, target)
            loss_count += loss.item()
            acc_count += acc
            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'train_loss = {loss_count / len(trainLoader):.2f}')
        print(f'train_accuracy = {acc_count / len(trainLoader):.2f}')

        # regular saving
        if epoch % 10 == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict()
            }
            save_file = os.path.join('./pth_save/folder3_seresnet12', 'ckpt_epoch_{epoch}.pth'.format(epoch = epoch))
            torch.save(state, save_file)
            
            
            model.eval()
            vacc_count = 0
            for vtarget, vinput in tqdm(valLoader):
                # ===================forward=====================
                vinput = vinput.reshape(-1, 3, 128, 128)
                vtarget = vtarget.reshape(-1, )
                output = model(vinput)
                vacc = count_acc(output, vtarget)
                vacc_count += vacc
            print(f'test_accuracy = {vacc_count / len(valLoader):.2f}')
            
            
    # save the last model
    state = {
        'model': model.state_dict()
    }
    save_file = os.path.join('./pth_save/folder3_seresnet12', 'seresnet12_last.pth')
    torch.save(state, save_file)
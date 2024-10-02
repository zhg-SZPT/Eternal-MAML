from collections import OrderedDict
from copy import deepcopy
from torch import nn, optim
import torch
import random
import argparse
import torch.nn.functional as F
from tqdm import tqdm
from category_split import *
from Data.mvtec import MvTec
import numpy as np
from utils import count_acc
from networks.resNet import resnet12, seresnet12, seresnet18, resnet50, seresnet50
from networks.se4conv import ConvNet



# def adjust_learning_rate(epoch, lr_decay_epochs = [20,100], learning_rate = 0.05, lr_decay_rate = 0.1):
#     """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
#     steps = np.sum(epoch > np.asarray(lr_decay_epochs))
#     if steps > 0:
#         new_lr = learning_rate * (lr_decay_rate ** steps)
#         return new_lr
#     else:
#         return learning_rate
    
            
def parse_option():
    parser = argparse.ArgumentParser('argument for training')
     # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--eval_epochs', type=int, default=50)
    parser.add_argument('--inner_iters', type=int, default=5)
    parser.add_argument('--batch_tasks', type=int, default=6)
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.5, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # setting for meta-learning
    parser.add_argument('--n_ways', type=int, default = 3, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--gd_lr', default=0.01, type=float,
                        help='The inner learning rate for MAML-Based model')
    parser.add_argument('--meta_lr', default=1e-3, type=float,
                        help='The inner learning rate for MAML-Based model')
    parser.add_argument('--temperature', default=0.5, type=float,
                        help='The inner learning rate for MAML-Based model')


    opt = parser.parse_args()
    return opt


# 参数池
opt = parse_option()

# 模型
# model = seresnet12(avg_pool=True, drop_rate=0.1, dropblock_size = 2, num_classes = opt.n_ways)
# 模型

model = seresnet12(avg_pool=True, drop_rate = 0.1, dropblock_size = 3, num_classes = opt.n_ways)

# model = ConvNet()
# model.fc = nn.Linear(64, opt.n_ways)
'''load feature extractor params'''
# pretrained_dic = torch.load('./pth_save/folder3_seresnet12/seresnet12_last.pth')['model']

# 过滤掉不匹配的权重（这里主要是最后一层）
# model_dic = model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dic.items() if 'classifier.' not in k}
# model_dic.update(pretrained_dict)
# model.load_state_dict(model_dic)

model.to('cuda')

#第一次初始化分类头
# fcone = nn.Linear(640, 1).to('cuda')
fcone = nn.Linear(640, 1).to('cuda')


#criterion/损失函数
criterion = nn.CrossEntropyLoss().to('cuda')
grad_lr = opt.gd_lr
# optimizer = optim.SGD(model.parameters(), lr= args.meta_lr) #元优化器
# classifier_parameters = model.classifier.parameters()

optimizer = optim.Adam(model.parameters(), lr = opt.meta_lr, weight_decay=opt.weight_decay)

#data
dataset = MvTec()
test_cat = random.sample(folder1['meta-test'], 3)
# test_cat = folder9['meta-test'][0]

# test_cat = ''
train_acc_pool = []
test_acc_pool = []
the_best_test_acc = 0
if __name__ == '__main__':
    # 训练
#     set cosine annealing scheduler
#     if opt.cosine:
#         eta_min = opt.meta_lr * (opt.lr_decay_rate ** 3)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)
        
    
    for epoch in range(1, opt.epochs + 1):
        
        print('==> Training...')
        print(f'--------epoch{epoch}--------')
        losses_q = [0 for _ in range(opt.inner_iters + 1)]  # losses_q[i] 是进行第 i 次更新后的 task_num 个 loss 值之和
        corrects = [0 for _ in range(opt.inner_iters + 1)]
        
        
        
        for _ in tqdm(range(opt.batch_tasks)):
            '''w init wc'''
#             model.fc.weight.data = fcone.weight.data.repeat(opt.n_ways, 1)
#             model.fc.bias.data = fcone.bias.data.repeat(opt.n_ways, 1).reshape(-1, )
            model.classifier.weight.data = fcone.weight.data.repeat(opt.n_ways, 1)
            model.classifier.bias.data = fcone.bias.data.repeat(opt.n_ways, 1).reshape(-1, )
            init_param = OrderedDict(model.named_parameters())
            support_data, support_label, query_data, query_label = dataset.sample(mode='train')
            support_data = support_data.to('cuda')
            query_data = query_data.to('cuda')
            support_label = torch.tensor(support_label, device='cuda').reshape(-1, )
            query_label = torch.tensor(query_label, device='cuda').reshape(-1, )

            # 1. 针对其中一个 Meta Task 进行第一次前向传播和反向传播
            logits = model.forward_fast_weights(support_data, init_param)  # 第一次前向传播
#             logits = logits / opt.temperature
            loss = F.cross_entropy(logits, support_label)
#             fast_weights, acc_gradients = model.update_params(loss, init_param, acc_gradients, step_size=opt.gd_lr, first_order=True)
            fast_weights = model.update_params(loss, init_param, step_size = grad_lr, first_order=True)


            # 2. 计算模型原始参数在query set上的loss和准确率
            with torch.no_grad():
                logitis_query = model.forward_fast_weights(query_data, init_param)
                loss = F.cross_entropy(logitis_query, query_label)
                losses_q[0] += loss.cpu()
                acc = count_acc(logitis_query, query_label, opt.n_ways)
                # print(acc)
                corrects[0] += acc

            with torch.no_grad():
                logitis_query = model.forward_fast_weights(query_data, fast_weights)
                loss = F.cross_entropy(logitis_query, query_label)
                losses_q[1] += loss.cpu()
                acc = count_acc(logitis_query, query_label, opt.n_ways)
                corrects[1] += acc

            for k in range(1, opt.inner_iters):
                # 1. 使用 support set 对模型进行参数更新
                logits = model.forward_fast_weights(support_data, fast_weights)  # 0.1GB\
#                 logits = logits / opt.temperature
                loss = F.cross_entropy(logits, support_label)
                # 3. 再次更新梯度，保存为 updated_params
#                 fast_weights, acc_gradients = model.update_params(loss, fast_weights, acc_gradients, step_size=opt.gd_lr, first_order=True)
                fast_weights = model.update_params(loss, fast_weights, step_size = grad_lr, first_order=True)

                # 2024/3/20
                if (k != opt.inner_iters - 1):

                    with torch.no_grad():
                        logits_q = model.forward_fast_weights(query_data, fast_weights)  # 0.5GB
                        
                        # 计算损失
                        loss_q = F.cross_entropy(logits_q, query_label)
                else:
                    logits_q = model.forward_fast_weights(query_data, fast_weights)
#                     logits_q = logits_q / opt.temperature
                    # 计算损失
                    loss_q = F.cross_entropy(logits_q, query_label)

                losses_q[k + 1] += loss_q.cpu()
                acc = count_acc(logits_q, query_label, opt.n_ways)
                corrects[k + 1] += acc
          
        # ===================outer_backward=====================
        loss_qq = losses_q[-1] # temperature
        optimizer.zero_grad()
        loss_qq.backward()
        
        
        weights_grad = model.classifier.weight.grad
        bias_grad = model.classifier.bias.grad
#         weights_grad = model.fc.weight.grad
#         bias_grad = model.fc.bias.grad
        
# #         gama = adjust_learning_rate(epoch)
        fcone.weight.data =  fcone.weight.data - opt.meta_lr * weights_grad.sum(dim = 0)
        fcone.bias.data = fcone.bias.data - opt.meta_lr * bias_grad.sum(dim = 0)
        
        optimizer.step()
        
        # 准确率
        accs = np.array(corrects) / opt.batch_tasks
        print('Epoch {}: Train Acc: {}'.format(epoch, accs))
        
        if epoch % 10 == 0:
            net = deepcopy(model)
#             net.fc.weight.data = fcone.weight.data.repeat(opt.n_ways, 1)
#             net.fc.bias.data = fcone.bias.data.repeat(opt.n_ways, 1).reshape(-1, )
            net.classifier.weight.data = fcone.weight.data.repeat(opt.n_ways, 1)
            net.classifier.bias.data = fcone.bias.data.repeat(opt.n_ways, 1).reshape(-1, )
            train_acc_pool.append(accs)
            print('==> Testing...')
            test_corrects = [0 for _ in range(opt.inner_iters + 1)]
            for cat in tqdm(test_cat):
                for i in range(1, opt.eval_epochs + 1):
                    '''w init wc'''
                    fast_test_weights = OrderedDict(net.named_parameters())
                    support_data, support_label, query_data, query_label = dataset.sample(mode='test', test_cat = cat)
                    support_data = support_data.to('cuda')
                    query_data = query_data.to('cuda')
                    support_label = torch.tensor(support_label, device='cuda').reshape(-1, )
                    query_label = torch.tensor(query_label, device='cuda').reshape(-1, )

                    with torch.no_grad():
                        logitis_query = net.forward_fast_weights(query_data, fast_test_weights)
                        acc = count_acc(logitis_query, query_label, opt.n_ways)
                        # print(acc)
                        test_corrects[0] += acc
                    for k in range(1, opt.inner_iters + 1):
                        logitis_s = net.forward_fast_weights(support_data, fast_test_weights)
#                         logitis_s = logitis_s / opt.temperature
                        s_loss = F.cross_entropy(logitis_s, support_label)
                        fast_test_weights = net.update_params(s_loss, fast_test_weights, step_size=grad_lr, first_order=True)

                        with torch.no_grad():
                            logitis_query = net.forward_fast_weights(query_data, fast_test_weights)
                            acc = count_acc(logitis_query, query_label, opt.n_ways)
                            # print(acc)
                            test_corrects[k] += acc
                            
            test_acc_pool.append(np.array(test_corrects) / (opt.eval_epochs * 3) )
            print('Test_Avg_Acc:  {}'.format(np.array(test_corrects) / (opt.eval_epochs * 3)))
            del net
            print(train_acc_pool[-1])
            print(test_acc_pool[-1])

































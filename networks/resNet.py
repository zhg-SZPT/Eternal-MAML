from collections import OrderedDict

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        # self.gamma = gamma
        # self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(planes, 4)

    def block_forward_para(self, x, params, base, mode, modules, downsample=False):
        '''the forard function of BasicBlock give parametes'''
        self.num_batches_tracked += 1

        residual = x

        out = F.conv2d(x, params[base + 'conv1.weight'], stride=(1, 1), padding=(1, 1))
        out = F.batch_norm(out, weight=params[base + 'bn1.weight'], bias=params[base + 'bn1.bias'],
                           running_mean=modules['bn1'].running_mean,
                           running_var=modules['bn1'].running_var, training=mode)
        out = self.relu(out)

        out = F.conv2d(out, params[base + 'conv2.weight'], stride=(1, 1), padding=(1, 1))
        out = F.batch_norm(out, weight=params[base + 'bn2.weight'], bias=params[base + 'bn2.bias'],
                           running_mean=modules['bn2'].running_mean,
                           running_var=modules['bn2'].running_var, training=mode)
        out = self.relu(out)

        out = F.conv2d(out, params[base + 'conv3.weight'], stride=(1, 1), padding=(1, 1))
        out = F.batch_norm(out, weight=params[base + 'bn3.weight'], bias=params[base + 'bn3.bias'],
                           running_mean=modules['bn3'].running_mean,
                           running_var=modules['bn3'].running_var, training=mode)
        if self.use_se:
            pre_out = out
            b, c, _, _ = out.size()
            out = F.adaptive_avg_pool2d(out, output_size=1).view(b,c)
            out = F.linear(out, weight=params[base+ 'se.fc.0.weight'], bias=params[base+ 'se.fc.0.bias'])
            out = F.relu(out)
            out = F.linear(out, weight=params[base + 'se.fc.2.weight'], bias=params[base + 'se.fc.2.bias'])
            out = torch.sigmoid(out).view(b,c,1,1)
            out = pre_out * out
            
        if downsample is True:
            residual = F.conv2d(x, params[base + 'downsample.0.weight'], stride=(1, 1))
            residual = F.batch_norm(residual, weight=params[base + 'downsample.1.weight'],
                                    bias=params[base + 'downsample.1.bias'],
                                    running_mean=modules['downsample']._modules['1'].running_mean,
                                    running_var=modules['downsample']._modules['1'].running_var, training=mode)
        out += residual
        out = F.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0,
                 dropblock_size=5, num_classes=-1, use_se=False):
        super(ResNet, self).__init__()

        self.inplanes = 3
        self.use_se = use_se
        self.layer1 = self._make_layer(block, n_blocks[0], 64,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160,
                                       stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(640, self.num_classes)


    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size, use_se=self.use_se)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x, is_feat = False):
        x = self.layer1(x)
        f0 = x
        x = self.layer2(x)
        f1 = x
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        f3 = x
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = x
        if self.num_classes > 0:
            x = self.classifier(x)
        if is_feat:
            return [f0, f1, f2, f3, feat], x
        else:
            return x

    def forward_fast_weights(self, x, fastweights):

        #layer 1
        x = self.layer1[0].block_forward_para(x, fastweights, 'layer1' + '.0.', self.training,
                                              self._modules['layer1']._modules['0']._modules, True)
#         x = self.layer1[1].block_forward_para(x, fastweights, 'layer1' + '.1.', self.training,
#                                               self._modules['layer1']._modules['1']._modules, False)
#         x = self.layer1[2].block_forward_para(x, fastweights, 'layer1' + '.2.', self.training,
#                                               self._modules['layer1']._modules['2']._modules, False)

        #layer 2

        x = self.layer2[0].block_forward_para(x, fastweights, 'layer2' + '.0.', self.training,
                                              self._modules['layer2']._modules['0']._modules, True)
#         x = self.layer2[1].block_forward_para(x, fastweights, 'layer2' + '.1.', self.training,
#                                               self._modules['layer2']._modules['1']._modules, False)
#         x = self.layer2[2].block_forward_para(x, fastweights, 'layer2' + '.2.', self.training,
#                                               self._modules['layer2']._modules['2']._modules, False)
#         x = self.layer2[3].block_forward_para(x, fastweights, 'layer2' + '.3.', self.training,
#                                               self._modules['layer2']._modules['3']._modules, False)

        #layer 3

        x = self.layer3[0].block_forward_para(x, fastweights, 'layer3' + '.0.', self.training,
                                              self._modules['layer3']._modules['0']._modules, True)
#         x = self.layer3[1].block_forward_para(x, fastweights, 'layer3' + '.1.', self.training,
#                                               self._modules['layer3']._modules['1']._modules, False)
#         x = self.layer3[2].block_forward_para(x, fastweights, 'layer3' + '.2.', self.training,
#                                               self._modules['layer3']._modules['2']._modules, False)
#         x = self.layer3[3].block_forward_para(x, fastweights, 'layer3' + '.3.', self.training,
#                                               self._modules['layer3']._modules['3']._modules, False)
#         x = self.layer3[4].block_forward_para(x, fastweights, 'layer3' + '.4.', self.training,
#                                               self._modules['layer3']._modules['4']._modules, False)
#         x = self.layer3[5].block_forward_para(x, fastweights, 'layer3' + '.5.', self.training,
#                                               self._modules['layer3']._modules['5']._modules, False)



        #layer 4

        x = self.layer4[0].block_forward_para(x, fastweights, 'layer4' + '.0.', self.training,
                                              self._modules['layer4']._modules['0']._modules, True)
#         x = self.layer4[1].block_forward_para(x, fastweights, 'layer4' + '.1.', self.training,
#                                               self._modules['layer4']._modules['1']._modules, False)
#         x = self.layer4[2].block_forward_para(x, fastweights, 'layer4' + '.2.', self.training,
#                                               self._modules['layer4']._modules['2']._modules, False)

        if self.keep_avg_pool:
            x = self.avgpool(x)

        x = x.view(x.size(0), -1)


        # Apply Linear Layer
        logits = F.linear(x, weight=fastweights['classifier.weight'], bias=fastweights['classifier.bias'])
        return logits
    
    #ANIL
#     def update_params(self, loss, params, step_size, first_order):
#         name_list, tensor_list = zip(*params.items())
#         grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
#         updated_params = OrderedDict()
#         # for name, param, grad in zip(name_list, tensor_list, grads):
#         for name, param, grad in zip(name_list, tensor_list, grads):
#             if 'classifier.' in name:
#                 #只更新分类头
#                 updated_params[name] = param - step_size * grad
#             else:
#                 updated_params[name] = param

#             return updated_params
    
    def update_params(self, loss, params, step_size, first_order):
        name_list, tensor_list = zip(*params.items())
        grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
        updated_params = OrderedDict()
        # for name, param, grad in zip(name_list, tensor_list, grads):
        for name, param, grad in zip(name_list, tensor_list, grads):
            updated_params[name] = param - step_size * grad
        return updated_params


def resnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def resnet18(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 2, 2], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def resnet24(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-24 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def resnet50(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-50 model.
    indeed, only (3 + 4 + 6 + 3) * 3 + 1 = 49 layers
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def resnet101(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-101 model.
    indeed, only (3 + 4 + 23 + 3) * 3 + 1 = 100 layers
    """
    model = ResNet(BasicBlock, [3, 4, 23, 3], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def seresnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


def seresnet18(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 2, 2], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


def seresnet24(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-24 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


def seresnet50(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-50 model.
    indeed, only (3 + 4 + 6 + 3) * 3 + 1 = 49 layers
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


def seresnet101(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-101 model.
    indeed, only (3 + 4 + 23 + 3) * 3 + 1 = 100 layers
    """
    model = ResNet(BasicBlock, [3, 4, 23, 3], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


if __name__ == '__main__':
    from collections import OrderedDict
    model = resnet50(avg_pool = True, drop_rate = 0.1, dropblock_size = 5, num_classes = 2).to('cuda')
    # print(model)
    init_param = OrderedDict(model.named_parameters())
    # # print(model)
    data = torch.randn(1, 3, 224, 224).to('cuda')
    logits = model.forward_fast_weights(data, init_param)
    print(logits)
    # label = torch.tensor([1,], device='cuda')
    # loss = F.cross_entropy(logits, label)
    # fast_weights = model.update_params(loss, init_param,step_size=0.8, first_order=True)
    # raise 0


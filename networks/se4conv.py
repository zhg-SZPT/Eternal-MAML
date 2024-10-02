from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch

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

class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, reduction=16, num_classes=10):
        super().__init__()
        self.num_layers = 4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # input layer
        self.add_module('{0}_{1}'.format(0,0), nn.Conv2d(x_dim, hid_dim, 3, padding=1))
        self.add_module('{0}_{1}'.format(0,1), nn.BatchNorm2d(hid_dim))
        self.add_module('se_{0}'.format(0), SELayer(hid_dim, reduction))
        # hidden layer
        for i in [1, 2]:
            self.add_module('{0}_{1}'.format(i,0), nn.Conv2d(hid_dim, hid_dim, 3, padding=1))
            self.add_module('{0}_{1}'.format(i,1), nn.BatchNorm2d(hid_dim))
            self.add_module('se_{0}'.format(i), SELayer(hid_dim, reduction))
        # last layer
        self.add_module('{0}_{1}'.format(3,0), nn.Conv2d(hid_dim, z_dim, 3, padding=1))
        self.add_module('{0}_{1}'.format(3,1), nn.BatchNorm2d(z_dim))
        self.add_module('se_{0}'.format(3), SELayer(z_dim, reduction))
        # fully connected layer
        self.add_module('fc', nn.Linear(z_dim, num_classes))

    def forward_fast_weights(self, x, params=None, embedding=False):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = x
        for i in range(self.num_layers):
            # Apply convolutional layer
            conv_weight = params['{0}_{1}.weight'.format(i, 0)]
            conv_bias = params['{0}_{1}.bias'.format(i, 0)]
            output = F.conv2d(output, conv_weight, bias=conv_bias, padding=1)

            # Apply batch normalization
            bn_weight = params['{0}_{1}.weight'.format(i, 1)]
            bn_bias = params['{0}_{1}.bias'.format(i, 1)]
            bn_running_mean = self._modules['{0}_{1}'.format(i, 1)].running_mean
            bn_running_var = self._modules['{0}_{1}'.format(i, 1)].running_var
            output = F.batch_norm(output, bn_running_mean, bn_running_var, weight=bn_weight, bias=bn_bias, training=self.training)

            # Apply ReLU activation
            output = F.relu(output)

            # Apply SE Layer
            se_layer = self._modules['se_{0}'.format(i)]
            b, c, _, _ = output.size()
            y = F.adaptive_avg_pool2d(output, 1).view(b, c)
            y = F.linear(y, weight=params['se_{0}.fc.0.weight'.format(i)], bias=params['se_{0}.fc.0.bias'.format(i)])
            y = F.relu(y, inplace=True)
            y = F.linear(y, weight=params['se_{0}.fc.2.weight'.format(i)], bias=params['se_{0}.fc.2.bias'.format(i)])
            y = torch.sigmoid(y).view(b, c, 1, 1)
            output = output * y

            # Apply max pooling
            output = F.max_pool2d(output, 2)

        output = self.avgpool(output)  # Apply adaptive average pooling
        output = output.view(x.size(0), -1)

        if embedding:
            return output
        else:
            # Apply fully connected layer
            fc_weight = params['fc.weight']
            fc_bias = params['fc.bias']
            logits = F.linear(output, weight=fc_weight, bias=fc_bias)
            return logits



    def update_params(self, loss, params, step_size=0.5, first_order=True):
        name_list, tensor_list = zip(*params.items())
        grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
        updated_params = OrderedDict()
        for name, param, grad in zip(name_list, tensor_list, grads):
            updated_params[name] = param - step_size * grad
        return updated_params

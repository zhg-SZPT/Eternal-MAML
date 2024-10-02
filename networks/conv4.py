import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.num_layers = 4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # input layer
        self.add_module('{0}_{1}'.format(0,0), nn.Conv2d(x_dim, hid_dim, 3, padding=1))   
        self.add_module('{0}_{1}'.format(0,1), nn.BatchNorm2d(hid_dim))
        # hidden layer
        for i in [1, 2]:
            self.add_module('{0}_{1}'.format(i,0), nn.Conv2d(hid_dim, hid_dim, 3, padding=1))   
            self.add_module('{0}_{1}'.format(i,1), nn.BatchNorm2d(hid_dim))     
        # last layer
        self.add_module('{0}_{1}'.format(3,0), nn.Conv2d(hid_dim, z_dim, 3, padding=1))   
        self.add_module('{0}_{1}'.format(3,1), nn.BatchNorm2d(z_dim))             
    
    def forward_fast_weights(self, x, params = None, embedding = False):
        if params is None:
            params = OrderedDict(self.named_parameters())
            
        output = x
        for i in range(self.num_layers):
            output = F.conv2d(output, params['{0}_{1}.weight'.format(i,0)], bias=params['{0}_{1}.bias'.format(i,0)], padding=1)
            output = F.batch_norm(output, weight=params['{0}_{1}.weight'.format(i,1)], bias=params['{0}_{1}.bias'.format(i,1)],
                                  running_mean=self._modules['{0}_{1}'.format(i,1)].running_mean,
                                  running_var=self._modules['{0}_{1}'.format(i,1)].running_var, training = self.training)
            output = F.relu(output)
            output = F.max_pool2d(output, 2)

        output = self.avgpool(output)     # AveragePool Here
        output = output.view(x.size(0), -1)
        
        if embedding:
            return output
        else:
            # Apply Linear Layer
            logits = F.linear(output, weight=params['fc.weight'], bias=params['fc.bias'])
            return logits

    def update_params(self, loss, params, step_size=0.5, first_order=True):
        name_list, tensor_list = zip(*params.items())
        grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
        updated_params = OrderedDict()
        for name, param, grad in zip(name_list, tensor_list, grads):
            updated_params[name] = param - step_size * grad
        return updated_params

if __name__ == '__main__':
    model = ConvNet()
    model.fc = nn.Linear(64, 2)
    criterion = nn.CrossEntropyLoss();
    updated_params = OrderedDict(model.named_parameters())
    data_query = torch.rand(size=(2, 3, 224, 224))
    data_label = torch.tensor([1,0]).reshape(-1,)
    output = model.forward_fast_weights(data_query, updated_params)
    loss = criterion(output, data_label)
    fast_params = model.update_params(loss, updated_params, step_size=0.2)
    # print(model)

    print(output)
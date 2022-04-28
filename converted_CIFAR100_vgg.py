import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import time

from CIFAR100_VGG16 import evaluate_accuracy


max_act = []
gamma = 2


def hook(module, input, output):
    '''
    use hook to easily get the maximum of each layers based on one training batch
    '''
    out = output.detach()
    out[out>1] /= gamma
    sort, _ = torch.sort(out.view(-1).cpu())
    max_act.append(sort[int(sort.shape[0] * 0.999) - 1])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        hooks = []
        cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))

        self.conv = cnn
        self.fc = nn.Linear(512, 100, bias=True)

        for i in range(len(self.conv)):
            hooks.append(self.conv[i].register_forward_hook(hook))
        hooks.append(self.fc.register_forward_hook(hook))
        self.hooks = hooks

    def forward(self, input):
        conv = self.conv(input)
        x = conv.view(conv.shape[0], -1)
        output = self.fc(x)
        return output


#=========== Do weight Norm, replace ReLU with SNode =============#
class SNode(nn.Module):
    def __init__(self, smode=True, gamma=5):
        super(SNode, self).__init__()
        self.smode = smode
        self.mem = 0
        self.spike = 0
        self.sum = 0
        self.threshold = 1.0
        self.opration = nn.ReLU(True)
        self.rsum = []
        self.summem = 0
        self.rmem = []
        self.gamma = gamma

    def forward(self, x):
        if not self.smode:
            out = self.opration(x)
        else:
            self.mem = self.mem + x

            self.spike = (self.mem / self.threshold).floor().clamp(min=0, max=self.gamma)
            self.mem = self.mem - self.spike
            out = self.spike
        return out


class SMaxPool(nn.Module):
    '''
    use lateral_ini to make output equal to the real value
    '''
    def __init__(self, smode=True, lateral_inhi=False):
        super(SMaxPool, self).__init__()
        self.smode = smode
        self.lateral_inhi = lateral_inhi
        self.sumspike = None
        self.opration = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sum = 0
        self.input = 0

    def forward(self, x):
        if not self.smode:
            out = self.opration(x)
        elif not self.lateral_inhi:
            self.sumspike += x
            single = self.opration(self.sumspike * 1000)
            sum_plus_spike = self.opration(x + self.sumspike * 1000)
            out = sum_plus_spike - single
        else:
            self.sumspike += x
            out = self.opration(self.sumspike)
            self.sumspike -= F.interpolate(out, scale_factor=2, mode='nearest')
        return out


def fuse_norm_replace(m, max_activation, last_max, smode=True, gamma=5, data_norm=True, lateral_inhi=False):
    '''
    merge conv and bn, then do data_norm
    :param m:                model
    :param max_activation:   the max_activation values on one training batch
    :param last_max:         the last max
    :param smode:            choose to use spike
    :param data_norm:
    :param lateral_inhi:
    :return:                 snn
    '''
    global index
    children = list(m.named_children())
    c, cn = None, None

    for i, (name, child) in enumerate(children):
        ind = index
        if isinstance(child, nn.Linear):
            if data_norm:
                child.weight.data /= max_activation[index] / max_activation[index-2]
                child.bias.data /= max_activation[index]
                last_max = max_activation[index]
        elif isinstance(child, nn.BatchNorm2d):
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = torch.nn.Identity()
            if data_norm:
                m._modules[cn].weight.data /= max_activation[index] / last_max
                m._modules[cn].bias.data /= max_activation[index]
                last_max = max_activation[index]
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        elif isinstance(child, nn.ReLU):
            m._modules[name] = SNode(smode=smode, gamma=gamma)
            if not data_norm:
                m._modules[name].threshold = max_activation[index]
                last_max = max_activation[index]
        elif isinstance(child, nn.MaxPool2d):
            m._modules[name] = SMaxPool(smode=smode, lateral_inhi=lateral_inhi)
        elif isinstance(child, nn.AvgPool2d):
            pass
        else:
            fuse_norm_replace(child, max_activation, last_max, smode, gamma, data_norm, lateral_inhi)
            index -= 1
        index += 1


def fuse(conv, bn):
    '''
    fuse the conv and bn layer
    '''
    w = conv.weight
    mean, var_sqrt, beta, gamma = bn.running_mean, torch.sqrt(bn.running_var + bn.eps), bn.weight, bn.bias
    b = conv.bias if conv.bias is not None else mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def clean_mem_spike(m):
    '''
    when change batch, you should clean the mem and spike of last batch
    :param m:  snn
    :return:
    '''
    children = list(m.named_children())
    for name, child in children:
        if isinstance(child, SNode):
            child.mem = 0
            child.spike = 0
        elif isinstance(child, SMaxPool):
            child.sumspike = 0
        else:
            clean_mem_spike(child)


def evaluate_snn(test_iter, snn, net, device=None, duration=50, plot=False, linetype=None):
    linetype = '-' if linetype==None else linetype
    accs = []
    acc_sum, n = 0.0, 0
    snn.eval()

    for test_x, test_y in tqdm(test_iter):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        n = test_y.shape[0]
        out = 0
        with torch.no_grad():
            clean_mem_spike(snn)
            acc = []
            for t in range(duration):
                start = time.time()
                out += snn(test_x)
                result = torch.max(out, 1).indices
                result = result.to(device)
                acc_sum = (result == test_y).float().sum().item()
                acc.append(acc_sum / n)
        accs.append(np.array(acc))

    accs = np.array(accs).mean(axis=0)

    print(max(accs))
    if plot:
        plt.plot(list(range(len(accs))), accs, linetype)
        plt.ylabel('Accuracy')
        plt.xlabel('Time Step')
        # plt.show()
        plt.savefig('./result.jpg')


if __name__ == '__main__':
    global index
    device = torch.device("cuda:1") if torch.cuda.is_available() else 'cpu'

    batch_size = 128
    normalize = torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    cifar100_train = datasets.CIFAR100(root='./data/', train=True, download=False, transform=transform_train)
    cifar100_test = datasets.CIFAR100(root='./data/', train=False, download=False, transform=transform_test)
    train_iter = torch.utils.data.DataLoader(cifar100_train, batch_size=200, shuffle=True, num_workers=0)  # 1024
    test_iter = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, num_workers=0)

    # the result of ANN
    net = CNN()
    net1 = deepcopy(net)
    [net1.hooks[i].remove() for i in range(len(net1.hooks))]
    net1.load_state_dict(torch.load("./saved_model/CIFAR100_VGG16_max.pth", map_location=torch.device(device)))
    net1 = net1.to(device)
    acc = evaluate_accuracy(test_iter, net1, device)
    print("acc on ann is : {:.4f}".format(acc))

    # get max activation on one training batch
    net2 = deepcopy(net)
    net2.load_state_dict(torch.load("./saved_model/CIFAR100_VGG16_max.pth", map_location=torch.device(device)))
    net2 = net2.to(device)
    _ = evaluate_accuracy(train_iter, net2, device, only_onebatch=True)
    # print(len(max_act))
    [net2.hooks[i].remove() for i in range(len(net2.hooks))]

    # data_norm
    net3 = deepcopy(net2)
    index = 0
    fuse_norm_replace(net3, max_act, last_max=1.0, smode=False, data_norm=True)

    index = 0
    fuse_norm_replace(net2, max_act, last_max=1.0, smode=True, gamma=gamma, data_norm=True, lateral_inhi=False)
    evaluate_snn(test_iter, net2, net3, device=device, duration=256, plot=True, linetype=None)




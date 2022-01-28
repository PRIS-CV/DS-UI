from __future__ import print_function
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import os
import argparse
import logging
from gmm_layer import MoMLayer
from model_resnet import ResidualNet
from uncertainty_measurements import *

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='None')
parser.add_argument("--savepath",    default='.', type=str)
parser.add_argument("--repeattimes", default='1', type=str)
parser.add_argument("--card",        default="0", type=str)
parser.add_argument("--n_component", default=1,   type=int)
args = parser.parse_args()

save_path = args.savepath + '/' + str(args.n_component) + '_' + args.repeattimes
if not os.path.exists(save_path):
    os.mkdir(save_path)

os.environ["CUDA_VISIBLE_DEVICES"] = args.card
nb_epoch = 100
lr = 7.5e-4
cycle_len = 70

print("OPENING " + save_path + '/results_train.csv')
print("OPENING " + save_path + '/results_test.csv')

results_train_file = open(save_path + '/results_train.csv', 'w')
results_train_file.write('epoch,train_acc,train_loss\n')
results_train_file.flush()

results_test_file = open(save_path + '/results_test.csv', 'w')
results_test_file.write('epoch,test_acc\n')
results_test_file.flush()

use_cuda = torch.cuda.is_available()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

uiset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
uiloader = torch.utils.data.DataLoader(uiset, batch_size=256, shuffle=False, num_workers=2)

# Model
net = ResidualNet("CIFAR10", 18)
class model_bn(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(model_bn, self).__init__() 
        self.features_1 = net
        self.num_ftrs = 512*1*1
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Dropout(0.7),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.7),
            nn.Linear(feature_size, feature_size),
            nn.BatchNorm1d(feature_size),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.7),
            )
        self.mom = MoMLayer(feature_size, classes_num, n_component=args.n_component, leaky=0.2)

    def forward(self, x):
        x = self.features_1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.mom(x)
        return x

classes_num = 10
net = model_bn(net, 2048, classes_num)


device = torch.device("cuda")

net = net.to(device)

net.features_1.to(device)
net.classifier.to(device)
net.mom.to(device)
net.features_1 = torch.nn.DataParallel(net.features_1)
net.classifier = torch.nn.DataParallel(net.classifier)
net.mom = torch.nn.DataParallel(net.mom)
#cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)

# Training
alpha = 1.
beta = 1e-4
gamma = 1.
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    correct2 = 0
    total = 0
    idx = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, outputs_gmm = net(inputs)

        if epoch < 1:
            loss = criterion(outputs, targets)
            net.mom.module.update_omega(targets)
        else:
            loss = criterion(outputs, targets) \
                 + alpha * net.mom.module.get_loss(outputs_gmm, targets) + beta * net.mom.module.get_reg_sigma()

        loss.backward()
        optimizer.step()

        train_loss += loss.detach().cpu()
        _, predicted = torch.max(outputs, -1)
        correct += predicted.eq(targets.data).sum().item()
        _, predicted = torch.max(outputs_gmm, -1)
        correct2 += predicted.eq(targets.data).sum().item()
        total += targets.size(0)
        

    train_acc = 100. * correct / total
    train_acc2 = 100. * correct2 / total
    train_loss = train_loss / (idx + 1)
    logging.info('Iteration %d, train_acc_cls = %.4f, train_acc_gmm = %.4f, train_loss = %.4f' % (epoch, train_acc, train_acc2, train_loss))
    results_train_file.write('%d,%.4f,%.4f,%.4f\n' % (epoch, train_acc, train_acc2, train_loss))
    results_train_file.flush()
    return train_acc, train_loss

def test(epoch):
    net.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    total = 0
    idx = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, outputs_gmm = net(inputs)
        outputs_gmm = torch.exp(outputs_gmm)

        loss = criterion(outputs, targets)

        test_loss += loss.detach().cpu()
        total += targets.size(0)
        _, predicted1 = torch.max(outputs, -1)
        correct1 += predicted1.eq(targets.data).sum().item()
        _, predicted2 = torch.max(outputs_gmm, -1)
        correct2 += predicted2.eq(targets.data).sum().item()

    test_acc1 = 100. * correct1 / total
    test_acc2 = 100. * correct2 / total
    test_loss = test_loss / (idx + 1)
    logging.info('test, test_acc_cls = %.4f, test_acc_gmm = %.4f, test_loss = %.4f' % (test_acc1, test_acc2, test_loss))
    results_test_file.write('%d,%.4f,%.4f,%.4f\n' % (epoch, test_acc1, test_acc2, test_loss))
    results_test_file.flush()
    return test_acc1, test_acc2, test_loss

def gmm_uncertainty():
    net.eval()
    correct1 = 0
    correct2 = 0
    total = 0
    idx = 0

    probs1_list = []
    alphas1_list = []
    probs2_list = []
    alphas2_list = []

    targets_list = []
    for batch_idx, (inputs, targets) in enumerate(uiloader):
        inputs = inputs.to(device)
        outputs, outputs_gmm = net(inputs)
        omega = net.mom.module.omega.data

        probs1_list += list(F.softmax(outputs, dim=1).data.cpu().numpy())
        alphas1_list += list(torch.exp(outputs).data.cpu().numpy())
        probs2_list += list(F.softmax(outputs_gmm, dim=1).data.cpu().numpy())
        alphas2_list += list(torch.exp(outputs_gmm).data.cpu().numpy() + 1.)
        targets_list += list(targets.data.numpy())

        targets = targets.to(device)
        total += targets.size(0)
        _, predicted1 = torch.max(outputs, -1)
        correct1 += predicted1.eq(targets.data).sum().item()
        _, predicted2 = torch.max(outputs_gmm, -1)
        correct2 += predicted2.eq(targets.data).sum().item()

    test_acc1 = 100. * correct1 / total
    test_acc2 = 100. * correct2 / total
    logging.info('test, test_acc_cls = %.4f, test_acc_gmm = %.4f' % (test_acc1, test_acc2))
    print('###################\n\n')

    probs1 = np.array(probs1_list)
    alphas1 = np.array(alphas1_list)
    probs2 = np.array(probs2_list)
    alphas2 = np.array(alphas2_list)
    targets = np.array(targets_list)
    print('1. fc output')
    aucs = gmm_auc(probs1, alphas1, targets)
    # aucs = (auroc_max_prob, auroc_ent, aupr_max_prob, aupr_ent)
    print('AUROC of Max.P/Ent.: %.4f %.4f' % aucs[:2])
    print('AUPR  of Max.P/Ent.: %.4f %.4f' % aucs[2:])
    print('###################\n\n')

    print('2. gmm output')
    aucs = gmm_auc(probs2, alphas2, targets)
    print('AUROC of Max.P/Ent.: %.4f %.4f' % aucs[:2])
    print('AUPR  of Max.P/Ent.: %.4f %.4f' % aucs[2:])
    print('###################\n\n')

def cycle_learning_rate(init_lr, cycle_len, n_epoch):
    lr = []
    for i in range(cycle_len // 2):
        lr += [init_lr / 10. + (init_lr - init_lr / 10.) / (cycle_len / 2) * i]
    for i in range(cycle_len - cycle_len // 2):
        lr += [init_lr / 10. + (init_lr - init_lr / 10.) / (cycle_len / 2) * (cycle_len / 2 - i)]
    for i in range(n_epoch - cycle_len):
        lr += [init_lr / 10. - (init_lr / 10. - 1e-6) / (n_epoch - cycle_len) * i]
    return lr
lr = cycle_learning_rate(lr, cycle_len, nb_epoch)

if os.path.exists(save_path + '/checkpoint.pth'):
    net = torch.load(save_path + '/checkpoint.pth')
else:
    max_val_acc = 0
    for epoch in range(nb_epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr[epoch]
            print(param_group['lr'])

        train(epoch)
        if epoch < 1:
            print(net.mom.module.omega.data.cpu().numpy().tolist())
        acc = test(epoch)
        val_acc = acc[0]
        torch.save(net, save_path + '/checkpoint.pth')

print('Uncertainty')
gmm_uncertainty()

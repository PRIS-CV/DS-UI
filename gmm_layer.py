import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

class MoMLayer(nn.Module):
    def __init__(self, num, num_out, n_component=1, leaky=0.2):
        super(MoMLayer, self).__init__()
        self.num = num
        self.num_out = num_out
        self.n_component = n_component
        self.leaky = leaky
        self.activation = nn.LeakyReLU(leaky, inplace=True)

        self.mu     = Parameter(torch.Tensor(   self.num, self.n_component, self.num_out))
        self.sigma  = Parameter(torch.zeros([1, self.num, self.n_component, self.num_out]))# log covariance matrix
        if n_component == 1:
            self.pi = torch.ones([1, 1, self.num_out]).cuda()# unormalized weights
        else:
            self.pi = Parameter(torch.ones([1,            self.n_component, self.num_out]))# unormalized weights

        self.omega = torch.zeros([1, self.num_out]).cuda()
        self.num_data = 0
        self.c = float(math.log(2 * math.pi) * self.num)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.mu, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(self.bias, -bound, bound)

    def log_gaussian_prob(self, x, mu, log_sigma):
        # return (torch.sum(log_sigma, dim=1) + torch.sum((x - mu).pow(2) / torch.exp(log_sigma), dim=1) + self.c).squeeze() / (-2)
        return (torch.sum(log_sigma, dim=1) + torch.sum((x - mu).pow(2) / torch.exp(log_sigma), dim=1)) / (-2.)

    def kl_mvn(self, log_sigma1, sigma2):
        # diagonal assumption for sigma
        # log_sigma1, sigma2: [n_c x n x m]
        # n: feature dimension (self.num), m: component number (self.num_out), n_c: MoM component number (self.n_component)
        # output: [n_c x m]
        return (torch.sum(torch.log(sigma2), dim=1) - torch.sum(log_sigma1, dim=1)) / 2.

    def kl_gmm_ub(self, pi1, log_sigma1, sigma2):
        # upper bound of kl divengence from gmm1 to gmm2, diagonal assumption for sigma
        # pi1: [n_c x m], log_sigma1, sigma2: [n_c x n x m]
        # n: feature dimension (self.num), m: component number (self.num_out), n_c: MoM component number (self.n_component)
        # output: [m x 1]
        return (pi1 * self.kl_mvn(log_sigma1, sigma2)).sum(dim=0).view(-1, 1)

    def get_reg_sigma(self, prior_sigma=None):
        # prior_sigma: [n_c x n x m]
        # n: feature dimension (self.num), m: component number (self.num_out), n_c: MoM component number (self.n_component)
        if prior_sigma == None:
            prior_sigma = torch.ones([self.n_component, self.num, self.num_out])
        return self.omega.matmul(self.kl_gmm_ub(F.softmax(self.pi, dim=1).view(self.n_component, self.num_out), \
                                                self.sigma.view(self.n_component, self.num, self.num_out), \
                                                prior_sigma.cuda()))

    def update_omega(self, targets):
        self.omega *= self.num_data
        for i in targets:
            self.omega[0, i.long()] += 1.
        self.num_data += targets.size(0)
        self.omega /= self.num_data

    def get_loss(self, outputs, targets):
        b = targets.size(0)
        one_hot_label = torch.zeros_like(outputs).cuda() - 1.# [b x c]
        one_hot_label[torch.arange(b), targets] = torch.ones([b]).cuda()
        one_hot_label = one_hot_label.cuda()
        return torch.mean(0. - torch.sum(one_hot_label.mul(F.softmax(outputs, dim=1)), dim=1))

    def forward(self, x):
        b = x.size(0)
        y = (self.activation(x).matmul(self.mu.view(self.num, -1)).view(b, self.n_component, self.num_out) \
            * self.pi.repeat(b, 1, 1)).sum(dim=1)
        z = x.view(b, -1, 1, 1).repeat(1, 1, self.n_component, self.num_out)
        z = F.softmax(self.pi, dim=1).repeat(b, 1, 1) \
            * self.log_gaussian_prob(z, self.mu.unsqueeze(0).expand_as(z), self.sigma.expand_as(z)).exp()
        z = self.omega.add_(1e-8).log().repeat(b, 1) + z.sum(dim=1).add_(1e-8).log()
        return y, z

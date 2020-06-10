import os
import yaml
import math
import random
import pickle
from time import time
from pprint import pprint
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

import nmslib

import numpy as np

from scipy import stats

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
from torchvision import models

from utils import *

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')


class MetaModel(object):
    """
    Holds model parameters while actual model is updated
    and can then restore the original parameters back to the model
    """
    def __init__(self, config, model, optimizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.momentum = self.config['model']['momentum']
        self.dampening = self.config['model']['dampening']
        self.nesterov = self.config['model']['nesterov']
        self.weight_decay = self.config['model']['wd']
        self.params = OrderedDict(self.model.named_parameters())

    def update(self, lr=0.1):
        for param_name in self.params.keys():
            param_state = self.optimizer.state[self.params[param_name]]
            path = param_name.split('.')
            cursor = self.model
            for module_name in path[:-1]:
                cursor = cursor._modules[module_name]
            if lr > 0:
                g = self.get_grad(param_state, self.params[param_name], self.params[param_name].grad)
                cursor._parameters[path[-1]] = self.params[param_name] - lr * g
            else:
                cursor._parameters[path[-1]] = self.params[param_name]

    def get_grad(self, state, p, g):
        d_p = g
        if self.weight_decay != 0:
            d_p = d_p.add(self.weight_decay * p.data)
        if self.momentum != 0:
            if 'momentum_buffer' not in state:
                buf = d_p
            else:
                buf = self.momentum * state['momentum_buffer'] + (1 - self.dampening) * d_p
            if self.nesterov:
                d_p = d_p.add(self.momentum, buf)
            else:
                d_p = buf
        return d_p

    def restore(self):
        self.update(lr=0)


class Classifier(nn.Module):
    """
    Parent class for classifiers which computes weighted and unweighted losses
    """
    def __init__(self):
        super().__init__()

    def loss(self, y_pred, y_true, w=None, reduction='mean'):
        if w is not None:
            if reduction == 'mean':
                return (F.nll_loss(y_pred, y_true, reduction='none') * w).mean()
            if reduction == 'sum':
                return (F.nll_loss(y_pred, y_true, reduction='none') * w).sum()
        else:
            if reduction == 'mean':
                return F.nll_loss(y_pred, y_true, reduction='none').mean()
            if reduction == 'sum':
                return F.nll_loss(y_pred, y_true, reduction='none').sum()


class LogisticRegression(Classifier):
    """
    Logistic regression model for computing downstream test accuracy from extracted features
    """
    def __init__(self, config, input_dim, num_classes):
        super().__init__()

        self.config = config
        self.input_dim = input_dim
        self.num_classes = num_classes

        if 'C' in self.config['data']['test']['logistic_regression']:
            C = self.config['data']['test']['logistic_regression']['C']
        else:
            C = 1 / (100. / (self.config['model']['hidden_dim'] * self.config['data']['test']['num_classes']))
        max_iter = self.config['data']['test']['logistic_regression']['max_iter']
        print('Logistic regression:', flush=True)
        print(f'\t solver = L-BFGS', flush=True)
        print(f'\t C = {C}', flush=True)
        print(f'\t max_iter = {max_iter}', flush=True)
        self.clf = LogReg(C=C, max_iter=max_iter, solver='lbfgs', multi_class='multinomial')

    def fit(self, data_loader, model, silent=True):
        batch_size = self.config['data']['test']['batch_size']

        all_features = torch.zeros(len(data_loader) * batch_size, self.config['model']['hidden_dim'])
        all_targets = torch.zeros(len(data_loader) * batch_size, dtype=torch.long)
        for i, (data, targets) in enumerate(data_loader):
            data, targets = data.to(self.config['device']), targets.to(self.config['device'])
            features = model.extract_features(data)
            all_features[batch_size * i:batch_size * (i + 1)] = features.detach().cpu()
            all_targets[batch_size * i:batch_size * (i + 1)] = targets.cpu()

        print('Training logistic regression', flush=True)
        self.clf.fit(all_features, all_targets)
        print(f"Validation accuracy: {100. * self.clf.score(all_features, all_targets):.2f}%", flush=True)

    def evaluate(self, data_loader, model, silent=True):
        loss, acc, acc_top5 = 0, 0, 0
        clf_acc = 0
        num_batches = 0
        with torch.no_grad():
            pbar = data_loader if silent else tqdm(data_loader, desc='Evaluating classifier on test data')
            for i, (data, targets) in enumerate(pbar):
                num_batches += 1
                data, targets = data.to(self.config['device']), targets.to(self.config['device'])
                features = model.extract_features(data)

                clf_acc_ = self.clf.score(features.cpu(), targets.cpu())
                clf_acc += clf_acc_
                if not silent:
                    pbar.set_postfix(test_acc=100. * clf_acc_)

        loss = None
        acc = (100. * clf_acc) / num_batches
        print(f"Test acc: {acc:.2f}%", flush=True)
        return loss, acc, None


class ResNet(Classifier):
    """
    Residual network architecture for ImageNet/VD experiments
    """
    def __init__(self, config, num_classes=1000, feature_layer=-1):
        super().__init__()

        self.config = config

        pretrained = True if 'pretrained' in self.config['model'] and self.config['model']['pretrained'] else False
        model = models.__dict__[self.config['model']['arch']](pretrained=pretrained, num_classes=1000)
        model.fc = nn.Linear(list(model.children())[-1].in_features, num_classes)
        modules = list(model.children())
        modules.insert(-1, nn.Flatten())
        self.feature_extractor = nn.Sequential(*modules[:feature_layer])
        self.classifier = nn.Sequential(*modules[feature_layer:])

        self.output_shapes = []
        x = torch.zeros(1, self.config['data']['channels'],
            self.config['data']['cropsize'], self.config['data']['cropsize'])
        for i, m in enumerate(modules):
            x = m(x)
            self.output_shapes.append(x.shape[1:])

        self.config['model']['hidden_dim'] = np.prod(self.output_shapes[feature_layer - 1])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return output

    def extract_features(self, x):
        x = self.feature_extractor(x)
        return torch.flatten(x, start_dim=1, end_dim=-1)


class AutoEncoder(nn.Module):
    """
    Parent class for autoencoders which computes weighted and unweighted losses
    """
    def __init__(self):
        super().__init__()

    def kld_loss(self, mu, logvar, weights, reduction='mean'):
        if weights is not None:
            if reduction == 'mean':
                return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1) * weights).mean()
            if reduction == 'sum':
                return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1) * weights).sum()
        else:
            if reduction == 'mean':
                return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            if reduction == 'sum':
                return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).sum()

    def recon_loss(self, x_recon, x, weights, reduction='mean'):
        if weights is not None:
            if reduction == 'mean':
                return (F.mse_loss(x, x_recon, reduction='none').sum(dim=-1) * weights).mean()
            if reduction == 'sum':
                return (F.mse_loss(x, x_recon, reduction='none').sum(dim=-1) * weights).sum()
        else:
            if reduction == 'mean':
                return F.mse_loss(x, x_recon, reduction='none').sum(dim=-1).mean()
            if reduction == 'sum':
                return F.mse_loss(x, x_recon, reduction='none').sum(dim=-1).sum()
    
    def loss(self, x_recon, x, mu, logvar, weights=None, reduction='mean'):
        loss = self.recon_loss(x_recon, x.view(x.shape[0], -1), weights, reduction)
        loss += self.beta * self.kld_loss(mu, logvar, weights, reduction)
        return loss


class MLPVAE(AutoEncoder):
    """
    Minimal variational autoencoder used in MNIST/FashionMNIST/KMNIST experiments
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.beta = self.config['model']['beta']

        self.fc1 = nn.Linear(self.config['data']['channels'] * (self.config['data']['cropsize'] ** 2),
            100)
        self.fc21 = nn.Linear(100, self.config['model']['hidden_dim'])
        self.fc22 = nn.Linear(100, self.config['model']['hidden_dim'])
        self.fc3 = nn.Linear(self.config['model']['hidden_dim'], 100)
        self.fc4 = nn.Linear(100, self.config['data']['channels'] * (self.config['data']['cropsize'] ** 2))

    def encode(self, x):
        x = F.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(z))

    def forward(self, x):
        x = x.view(-1, self.config['data']['channels']
            * (self.config['data']['cropsize'] ** 2)) if self.config['data']['channels'] == 1 else x
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)
        y = y.view(-1, self.config['data']['channels'], self.config['data']['cropsize'],
            self.config['data']['cropsize']) if self.config['data']['channels'] == 3 else y
        return y, mu, logvar, z


def select_model(config):
    # set up model
    if config['data']['training']['task'] == 'rotation-prediction':
        if config['model']['arch'].startswith('resnet'):
            model = ResNet(config,
                          num_classes=config['data']['training']['num_classes'],
                          feature_layer=config['model']['feature_layer']
                          ).to(config['device'])
        else:
            print(f"Model architecture {config['model']['arch']} not found", flush=True)
    elif config['data']['training']['task'] == 'reconstruction':
        if config['model']['arch'] == 'mlp':
            model = MLPVAE(config).to(config['device'])
        else:
            print(f"Model architecture {config['model']['arch']} not found", flush=True)
    return model


class BaselineWeights(nn.Module):
    """
    Maintains a weight of 1 for each data point in the training set which is never updated.
    This class is used by the baseline VAE, Oracle-VAE and RotNet.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.weights = torch.ones(self.config.num_data_points)

    def get(self, indices):
        weight = self.weights[indices] # use indices to select weights
        #weight.requires_grad = True # and allow grads
        return weight

    def get_all_weights(self, train_loader=None, model=None):
        return self.weights.detach()


class kNNWeights(nn.Module):
    """
    Computes a weight based on euclidean distances for each data point in the training set.
    Used in our NN-Weighter implementation of Peng et al. ACML, 2019.
    Our version uses approximate nearest neighbour for efficiency on large datasets.
    """
    def __init__(self, config, index_data, query_data, k=1):
        self.config = config
        self.k = k
        self.distances_dir = os.path.join('../distances', self.config['data']['test_set'])
        if not os.path.isdir(self.distances_dir):
            os.makedirs(self.distances_dir)
        self.distances_path = os.path.join(self.distances_dir, 'distances.pt')
        if os.path.exists(self.distances_path):
            self.load_distances()
        else:
            self.compute_distances(index_data, query_data)
            self.save_distances()

        self.compute_weights()

    def compute_distances(self, index_data, query_data):
        print('Preparing nearest neighbor data')
        print(f'\t Number of fit data points {len(index_data)}')
        print(f'\t Number of predict data points {len(query_data)}')
        fit_loader = torch.utils.data.DataLoader(index_data, batch_size=len(index_data), shuffle=False, drop_last=False)
        print('Created fit loader')
        query_loader = torch.utils.data.DataLoader(query_data,
            batch_size=self.config['data']['training']['batch_size'], shuffle=False)
        print('Created predict loader')
        x_fit, _ = fit_loader.__iter__().next()
        x_fit = x_fit.flatten(start_dim=1)

        print('Creating ANN index')
        start = time()
        # initialize a new index, using a HNSW index on Euclidean space
        index = nmslib.init(method='hnsw', space='l2')
        index.addDataPointBatch(x_fit)
        index_time_params = {'M': 5, 'indexThreadQty': 8, 'efConstruction': 10, 'post' : 0}
        index.createIndex({'post': 2}, print_progress=True)
        print(f'Time taken: {time() - start:.2f}')

        print('Computing distances')
        start = time()
        dists = []
        for data, _, _ in tqdm(query_loader):
            data = data.flatten(start_dim=1)
            _, distances = tuple(zip(*index.knnQueryBatch(data.numpy(), k=self.k, num_threads=8)))
            dists.extend(distances)
        print(f'Time taken: {time() - start:.2f}')
        self.dists = torch.tensor(dists)

    def save_distances(self):
        torch.save(self.dists, self.distances_path)
        print(f'k-NN distances saved at {self.distances_path}')

    def load_distances(self):
        print(f'Loading k-NN distances from {self.distances_path}')
        self.dists = torch.load(self.distances_path)

    def compute_weights(self):
        print('Converting distances into weights')
        self.weights = 1 / (self.config['weights']['knn_beta'] * self.dists).exp()

    def get(self, indices):
        weight = self.weights[indices] # use indices to select weights
        #weight.requires_grad = True # and allow grads
        return weight

    def get_all_weights(self, train_loader=None, model=None):
        return self.weights.detach()


class Weights(nn.Module):
    """
    Maintains a weight for each data point in the training set.
    Used for the Deterministic DataWeighter as well as for L2RW
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.lr = config['weights']['lr']

        self.reset()

    def reset(self):
        self.weights = nn.Parameter(torch.ones(self.config.num_data_points) * self.config['weights']['init'])

        self.optimizer_params = [self.weights]
        if self.config['weights']['optimizer'] == 'sgd':
            opt = optim.SGD
        elif self.config['weights']['optimizer'] == 'adam':
            opt = optim.Adam
        self.optimizer = opt(self.optimizer_params, lr=self.lr, momentum=self.config['weights']['momentum'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, self.config['weights']['lr_gamma'])

    def set_to_one(self):
        self.weights = nn.Parameter(torch.ones(self.config.num_data_points) * 1)

    def normalise(self, w):
        if w.sum() == 0:
            return w
        else:
            return w / w.sum()

    def clamp_at_zero(self, w):
        return torch.clamp(w, min=0)

    def clamp_at_one(self, w):
        return torch.clamp(w, max=1)

    def get(self, indices):
        weight = self.weights[indices] # use indices to select weights
        return weight

    def get_all_weights(self, train_loader=None, model=None):
        return self.weights.detach()

    def update(self, weight, indices, loss):
        self.optimizer.zero_grad()
        self.weights.grad = torch.autograd.grad(loss, self.parameters())[0]
        nn.utils.clip_grad_norm_(self.weights, self.config['weights']['max_grad_norm'])
        self.optimizer.step()
        return self.weights.grad[indices].detach()


class BetaWeights(nn.Module):
    """
    Maintains a Beta distribution defined by (a, b) for each data point in the training set.
    Weights are sampled using the reparameterization trick to allow gradient updates.
    Initially set (a = 1, b = 1)
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.lr = config['weights']['lr']
        # init both a and b to ln(1)
        self.log_a = nn.Parameter(torch.ones(self.config.num_data_points) * torch.log(torch.ones(1)))
        self.log_b = nn.Parameter(torch.ones(self.config.num_data_points) * torch.log(torch.ones(1)))

        self.optimizer_params = [self.log_a, self.log_b]
        if self.config['weights']['optimizer'] == 'sgd':
            opt = optim.SGD
        elif self.config['weights']['optimizer'] == 'adam':
            opt = optim.Adam
        self.optimizer = opt(self.optimizer_params, lr=self.lr, momentum=self.config['weights']['momentum'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, self.config['weights']['lr_gamma'])

    def reset(self):
        self.log_a = nn.Parameter(torch.ones(self.config.num_data_points) * torch.log(torch.ones(1)))
        self.log_b = nn.Parameter(torch.ones(self.config.num_data_points) * torch.log(torch.ones(1)))

    def set_to_one(self):
        # makes all densities highly concentrated at 1
        self.log_a = nn.Parameter(torch.ones(self.config.num_data_points) * torch.log(torch.ones(1) * 20))
        self.log_b = nn.Parameter(torch.ones(self.config.num_data_points) * torch.log(torch.ones(1) * .001))

    def get(self, indices):
        # use reparameterization trick to get weights
        weight = Beta(self.log_a[indices].exp(), self.log_b[indices].exp()).rsample([1])
        return weight

    def get_all_weights(self, train_loader=None, model=None):
        return self.mean().detach()

    def cdf(self, x):
        c = stats.beta(self.log_a.exp().detach().numpy(), self.log_b.exp().detach().numpy()).cdf(x)
        return torch.tensor(c)

    def mean(self):
        return Beta(self.log_a.exp(), self.log_b.exp()).mean

    def variance(self, device=None):
        if device is not None:
            return Beta(self.log_a.to(device).exp(), self.log_b.to(device).exp()).variance
        else:
            return Beta(self.log_a.exp(), self.log_b.exp()).variance

    def update(self, weight, indices, loss):
        self.optimizer.zero_grad()
        self.log_a.grad, self.log_b.grad = torch.autograd.grad(loss, self.parameters())
        self.optimizer.step()
        return self.log_a.grad[indices].detach()


def select_weights(config, train_set, val_train_set):
    # set up weights
    if 'weights' in config.config_dict.keys():
        if 'arch' in config['weights'] and config['weights']['arch'] == 'beta-weights':
            weights = BetaWeights(config)
        elif 'arch' in config['weights'] and config['weights']['arch'] == 'knn-weights':
            weights = kNNWeights(config, val_train_set, train_set)
        else:
            weights = Weights(config)
    else:
        weights = BaselineWeights(config)
    return weights

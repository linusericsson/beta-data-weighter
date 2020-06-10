import os
from time import time
import random
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm as tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as FT

import torchvision

from models import *
from data import *
from utils import *


class Trainer():
    """
    This class sets up the whole experiment, loads the data, initialises the model and the instance weights
    and trains the model using one of the following algorithms: BDW, DW, L2RW, NNW or the baselines (RotNet, VAE, Oracle-VAE).
    """
    def __init__(self, config):
        self.config = config

        self.update_rules = {
            'baseline': baseline_update,            # the baseline update rule is used by NNW and the baselines
            'meta-gradient': meta_gradient_update   # the meta-gradient update rule is used by BDW, DW and L2RW
        }
        self.update = self.update_rules[self.config['trainer']['algorithm']]
        
        self.set_seeds()
        self.prepare_data()
        self.prepare_models()

        self.results = pd.DataFrame({
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        })

        self.classifier = None

        if self.config['data']['test']['task'] == 'reconstruction':
            self.best_measure = np.inf
        else:
            self.best_measure = -np.inf

    def prepare_data(self):
        self.data = Data(self.config)

    def prepare_models(self):
        self.weights = select_weights(self.config, self.data.train_set, self.data.val_train_set)
        self.model = select_model(self.config)

        print(f'Number of model parameters: {count_parameters(self.model)}', flush=True)
                
        if self.config['model']['optimizer'] == 'sgd':
            opt = optim.SGD
        elif self.config['model']['optimizer'] == 'adam':
            opt = optim.Adam
        self.optimizer = opt(self.model.parameters(), lr=self.config['model']['lr'],
            weight_decay=self.config['model']['wd'], momentum=self.config['model']['momentum'])

        if 'lr_milestones' in self.config['model']:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.config['model']['lr_milestones'],
                gamma=self.config['model']['lr_gamma'])
        else:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1,
                gamma=self.config['model']['lr_gamma'])

        # make meta model
        # the MetaModel object will save the original model params while we do some updates with them
        self.meta_model = MetaModel(self.config, self.model, self.optimizer)

    def set_seeds(self):
        random.seed(self.config['exp']['seed'])
        np.random.seed(self.config['exp']['seed'])
        torch.manual_seed(self.config['exp']['seed'])
        torch.cuda.manual_seed(self.config['exp']['seed'])

    def save_models(self, val_loss, val_acc, all_weights, force=False):
        # save models to file
        if self.config['data']['test']['task'] == 'reconstruction':
            if force or val_loss < self.best_measure:
                self.best_measure = val_loss
                print(f'Saving model with validation loss: {val_loss:.2f}')
                torch.save(self.model.state_dict(), os.path.join(self.config.models_path, 'model.pt'))
        else:
            if force or val_acc > self.best_measure:
                self.best_measure = val_acc
                print(f'Saving model with validation accuracy: {val_acc:.2f}%')
                torch.save(self.model.state_dict(), os.path.join(self.config.models_path, 'model.pt'))
        torch.save(all_weights, os.path.join(self.config.models_path, 'weights.pt'))

    def test(self, epoch, fit_loader, test_loader):
        self.model.eval()
        test_loss = 0
        test_acc, test_acc_top5 = 0, 0

        if self.config['data']['test']['task'] == 'logistic-regression':
            if self.classifier is None:
                self.classifier = LogisticRegression(self.config, self.config['model']['hidden_dim'],
                    self.config['data']['test']['num_classes']).to(self.config['device'])
            self.classifier.fit(fit_loader, self.model, silent=False)
            return self.classifier.evaluate(test_loader, self.model, silent=False)
        else:
            num_data_points = 0
            with torch.no_grad():
                for i, (data, target) in enumerate(test_loader):
                    num_data_points += data.size(0)
                    # rotate images
                    if self.config['data']['test']['task'] == 'rotation-prediction':
                        data, targets = rotate(data, c=self.config['data']['test']['num_classes'])

                    data, target = data.to(self.config['device']), target.to(self.config['device'])

                    if self.config['data']['test']['task'] == 'prototypical-networks':
                        # evaluate feature extractor using prototypical network
                        tl, ta = prototypical_network_loss(self.config, data, target, self.model, mode='test')
                        tl *= data.size(0)
                        ta *= data.size(0)
                        test_loss += tl.item()
                        test_acc += ta
                    elif self.config['data']['test']['task'] in ['rotation-prediction', 'classification']:
                        output = self.model(data)
                        tl = self.model.loss(output, target).item()
                        tl *= data.size(0)
                        test_loss += tl
                        if 'imagenet' in self.config['data']['test_set']:
                            ta, ta_top5 = accuracy(output, target, topk=(1, 5))
                            ta *= data.size(0)
                            ta_top5 *= data.size(0)
                            test_acc += ta
                            test_acc_top5 += ta_top5
                        else:
                            ta = 100. * count_acc(output, target)
                            ta *= data.size(0)
                            test_acc += ta
                    elif self.config['data']['test']['task'] == 'reconstruction':
                        output, mu, logvar, z = self.model(data)
                        tl = self.model.loss(output, data, mu, logvar).item()
                        tl *= data.size(0)
                        test_loss += tl
            test_loss /= num_data_points
            test_acc /= num_data_points
            test_acc_top5 /= num_data_points
            self.model.train()
            if 'imagenet' in self.config['data']['test_set']:
                return test_loss, test_acc, test_acc_top5
            else:
                return test_loss, test_acc, None

    def load_best_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.config.models_path, "model.pt")))

    def save_results(self, test_loss, test_acc, test_acc_top5):
        results = pd.DataFrame({
            'test_loss': [test_loss],
            'test_acc': [test_acc],
            'test_acc_top5': [test_acc_top5]
        })
        results.to_csv(os.path.join(self.config.results_path, 'test_results.csv'))
        
    def train(self):
        self.model.train()
        epoch = 0

        gpu_info(self.config)

        # evaluate and log results
        val_loss, val_acc, val_acc_top5 = self.test(epoch, self.data.test_train_loader, self.data.val_test_loader)
        self.results = self.results.append({
            'train_loss': None,
            'train_acc': None,
            'meta_loss': None,
            'meta_acc': None,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_acc_top5': val_acc_top5,
            'batch_time': None,
            'epoch_time': None,
            'proportion_of_data': 1.0
        }, ignore_index=True)
        print(self.results.tail(), flush=True)

        all_weights = self.weights.get_all_weights(self.data.train_loader, self.model)

        # save results, models and config
        self.results.to_csv(os.path.join(self.config.results_path, 'results.csv'))
        self.save_models(val_loss, val_acc, all_weights)
        pickle.dump(self.config, open(os.path.join(self.config.timestamp_path, 'config.pkl'), 'wb'))

        # start training
        for epoch in range(1, self.config['trainer']['num_epochs'] + 1):
            epoch_start = time()
            batch_times = []
            grads = torch.zeros(self.config.num_data_points)
            train_loss, train_acc, meta_loss, meta_acc = 0, 0, 0, 0
            pbar = tqdm(enumerate(self.data.train_loader), total=len(self.data.train_loader), desc=f'Epoch {epoch}')

            # reset the weights for the L2RW algorithm
            if 'weights' in self.config.config_dict.keys():
                if 'reset' in self.config['weights'] and self.config['weights']['reset']:
                    self.weights.reset()

            for batch, (data, targets, indices) in pbar:
                batch_start = time()

                # rotate images
                if self.config['data']['training']['task'] == 'rotation-prediction':
                    data, targets = rotate(data, c=self.config['data']['training']['num_classes'])

                data = data.to(self.config['device'])
                targets = targets.to(self.config['device'])

                # run the update procedure for this trainer
                tl, ta, vl, va, g = self.update(data, targets, indices, self.config, self.model, self.optimizer,
                    self.meta_model, self.weights, self.data.train_loader, self.data.val_train_loader)
                train_loss += tl
                if ta is not None:
                    train_acc += ta
                else:
                    train_acc = None
                if vl is not None:
                    meta_loss += vl
                else:
                    meta_loss = None
                if va is not None:
                    meta_acc += va
                else:
                    meta_acc = None

                if g is not None:
                    grads[indices] = g

                batch_end = time()
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)
                pbar.set_postfix(epoch=epoch, batch=f'{batch}/{len(self.data.train_loader)}',
                    train_loss=tl, train_acc=ta, meta_loss=vl, meta_acc=va, batch_time=f'{batch_time:.3f}')

            epoch_end = time()
            epoch_time = epoch_end - epoch_start

            train_loss /= len(self.data.train_loader)
            if train_acc is not None:
                train_acc /= len(self.data.train_loader)
            if meta_loss is not None:
                meta_loss /= len(self.data.train_loader)
            if meta_acc is not None:
                meta_acc /= len(self.data.train_loader)

            # anneal learning rates
            self.scheduler.step(epoch)
            if hasattr(self.weights, 'scheduler'):
                self.weights.scheduler.step(epoch)

            # get current weights for all training points
            all_weights = self.weights.get_all_weights(self.data.train_loader, self.model)

            # prune datapoints of low weight
            if ('weights' in self.config.config_dict.keys() and 'pruning' in self.config['weights']):
                proportion_left = self.data.prune(all_weights, grads, self.weights)
            else:
                proportion_left = 1.0

            gpu_info(self.config)

            # evaluate and log results
            val_loss, val_acc, val_acc_top5 = self.test(epoch, self.data.test_train_loader, self.data.val_test_loader)
            self.results = self.results.append({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'meta_loss': meta_loss,
                'meta_acc': meta_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_acc_top5': val_acc_top5,
                'batch_time': torch.tensor(batch_times).mean().item(),
                'epoch_time': epoch_time,
                'proportion_of_data': proportion_left
            }, ignore_index=True)
            print(self.results.tail(), flush=True)

            # save results, models and config
            self.results.to_csv(os.path.join(self.config.results_path, 'results.csv'))
            self.save_models(val_loss, val_acc, all_weights)
            pickle.dump(self.config, open(os.path.join(self.config.timestamp_path, 'config.pkl'), 'wb'))


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


def prototypical_network_loss(config, data, target, model, mode='validation'):
    c = config['data'][mode]['num_classes_per_batch'] if mode == 'validation' else config['data'][mode]['num_classes']
    p = config['data'][mode]['num_data_per_class'] * c

    # the batch consists of some prototype data points and some query data points
    data_shot, data_query = data[:p], data[p:]
    target_shot, target_query = target[:p], target[p:]

    # we pass the prototype data points through the model so we can get our embedded prototypes
    proto = model.extract_features(data_shot)

    # we want one prototype per class so we take the mean of all examples per class
    proto = proto.reshape(config['data'][mode]['num_data_per_class'], c, -1).mean(dim=0)

    # the order of the classes is irrelevant as long as its the same between the prototypes and the query data
    label = torch.arange(c).repeat(config['data'][mode]['num_queries']).to(config['device'])

    # compute the distances between the embedded query data and the embedded prototypes
    query = model.extract_features(data_query)
    logits = euclidean_metric(query, proto)

    # use these distances as the logits in the cross-entropy calculation
    loss = F.cross_entropy(logits, label)
    acc = 100. * count_acc(logits, label)

    return loss, acc


def baseline_update(data, targets, indices, config, model, optimizer, meta_model, weights, train_loader, val_train_loader):
    model.train()

    # get weights for this minibatch
    weight = weights.get(indices).to(config['device'])

    # we need to tile the weights to make up for the 4 versions of each image
    if config['data']['training']['task'] == 'rotation-prediction':
        weight = weight.repeat_interleave(config['data']['training']['num_classes']).view(1, -1)

    # get the losses for the data points
    if config['data']['training']['task'] in ['rotation-prediction', 'classification']:
        train_y = model(data)
        train_loss = model.loss(train_y, targets, weight)
    elif config['data']['training']['task'] == 'reconstruction':
        train_y, mu, logvar, z = model(data)
        train_loss = model.loss(train_y, data, mu, logvar, weight)

    # update the model
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if config['data']['training']['task'] in ['rotation-prediction', 'classification']:
        train_acc = 100. * count_acc(train_y, targets)
    else:
        train_acc = None

    return train_loss.item(), train_acc, None, None, None


def get_minibatch_weights(config, weights, indices):
    weight = weights.get(indices).to(config['device'])
    # we need to tile the weights to make up for the 4 versions of each image
    if config['data']['training']['task'] == 'rotation-prediction':
        weight = weight.repeat_interleave(config['data']['training']['num_classes']).view(1, -1)
    return weight


def meta_gradient_update(data, targets, indices, config, model, optimizer, meta_model, weights, train_loader, val_train_loader):
    model.train()
    for i in range(config['weights']['steps']):
        # get val batch
        meta_data, meta_targets = val_train_loader.__iter__().next()

        # rotate images
        if config['data']['validation']['task'] == 'rotation-prediction':
            meta_data, meta_targets = rotate(meta_data, c=config['data']['validation']['num_classes'])
        meta_data, meta_targets = meta_data.to(config['device']), meta_targets.to(config['device'])

        # get weights for this minibatch
        weight = get_minibatch_weights(config, weights, indices)

        # get the losses for the data points
        if config['data']['training']['task'] in ['rotation-prediction', 'classification']:
            train_y = model(data)
            if 'l2rw' in config['exp']['name']:
                train_loss = model.loss(train_y, targets, weight, reduction='sum')
            else:
                train_loss = model.loss(train_y, targets, weight)
        elif config['data']['training']['task'] == 'reconstruction':
            train_y, mu, logvar, z = model(data)
            if config['weights']['arch'] == 'shared':
                weight = weights(z)
            if 'l2rw' in config['exp']['name']:
                train_loss = model.loss(train_y, data, mu, logvar, weight, reduction='sum')
            else:
                train_loss = model.loss(train_y, data, mu, logvar, weight)

        # accumulate gradients for both model and meta model
        optimizer.zero_grad()
        train_loss.backward(create_graph=True)

        # update model, and store original model params in meta model
        meta_model.update(optimizer.param_groups[0]['lr'])

        # compute val loss
        if config['data']['validation']['task'] == 'prototypical-networks':
            # evaluate feature extractor using prototypical network
            meta_loss, meta_acc = prototypical_network_loss(config, meta_data, meta_targets, model)
        elif config['data']['validation']['task'] in ['rotation-prediction', 'classification']:
            meta_y = model(meta_data) # now using the new parameters from the meta model update
            meta_loss = model.loss(meta_y, meta_targets)
            meta_acc = 100. * count_acc(meta_y, meta_targets)
        elif config['data']['validation']['task'] == 'reconstruction':
            meta_y, meta_mu, meta_logvar, _ = model(meta_data) # now using the new parameters from the meta model update
            meta_loss = model.loss(meta_y, meta_data, meta_mu, meta_logvar)
            meta_acc = None

        # perform the updates to theta
        if not config['model']['fixed'] and i == config['weights']['steps'] - 1 and not config['model']['new_weights']:
            optimizer.step() # and now the accumulated gradients are used to make an actual model update

        meta_model.restore() # this restores the model to its original params, which were stored in meta model

        # perform the update step for the weights
        grads = weights.update(weight, indices, meta_loss)

        # get weights again for this minibatch
        weight = get_minibatch_weights(config, weights, indices)

        # optionally normalise
        if config['weights']['clamp_at_zero']:
            weight = weights.clamp_at_zero(weight)
        if config['weights']['clamp_at_one']:
            weight = weights.clamp_at_one(weight)
        if config['weights']['normalise']:
            previous_weight = weight.view(weight.size())
            weight = weights.normalise(weight)

        # optionally perform the model update using the new weights
        # this is what L2RW does
        if config['model']['new_weights']:
            optimizer.zero_grad()
            if config['data']['training']['task'] in ['rotation-prediction', 'classification']:
                if 'l2rw' in config['exp']['name']:
                    train_loss = model.loss(train_y, targets, weight, reduction='sum')
                else:
                    train_loss = model.loss(train_y, targets, weight)
            elif config['data']['training']['task'] == 'reconstruction':
                if config['weights']['arch'] == 'shared':
                    weight = weights(z)
                if 'l2rw' in config['exp']['name']:
                    train_loss = model.loss(train_y, data, mu, logvar, weight, reduction='sum')
                else:
                    train_loss = model.loss(train_y, data, mu, logvar, weight)

        if config['data']['training']['task'] in ['rotation-prediction', 'classification']:
            with torch.no_grad():
                train_acc = 100. * count_acc(train_y, targets)
        else:
            train_acc = None

    return train_loss.item(), train_acc, meta_loss.item(), meta_acc, grads


# Load a specific train_data file
def load_trainer(config, override_path=None):
    if override_path is not None:
        path_to_exp = override_path
    else:
        path_to_exp = config.results_path
    return pickle.load(open(os.path.join(path_to_exp, 'trainer.pkl'), 'rb'))

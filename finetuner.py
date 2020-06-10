import torch

import pandas as pd

from tqdm import tqdm

from config import Config
from models import *
from data import *
from utils import *


class FineTuner():
    def __init__(self, config, data):
        print('Finetuner created')

        self.config = config
        self.data = data

        self.episodes = self.config['finetuning']['episodes']

        print('Loading best saved model')
        self.model = self.load_model(self.config, load=self.config['finetuning']['load_model'])

        self.data.test_train_loader = torch.utils.data.DataLoader(self.data.val_train_set,
                                                           batch_size=self.config['finetuning']['batch_size'],
                                                           shuffle=True, drop_last=True)

        self.data.val_test_loader = torch.utils.data.DataLoader(self.data.val_test_set,
                                                           batch_size=self.config['finetuning']['batch_size'],
                                                           shuffle=True, drop_last=True)

        self.config['data']['test']['batch_size'] = self.config['finetuning']['batch_size']

        self.best_measure = -np.inf
        self.best_hyperparams = {}

    def save_results(self, test_loss, test_acc, test_acc_top5):
        results = pd.DataFrame({
            'test_loss': [test_loss],
            'test_acc': [test_acc],
            'test_acc_top5': [test_acc_top5]
        })
        results.to_csv(os.path.join(self.config.results_path, 'tuned_results.csv'))

    def save_model(self, measure, epoch, tune_params=None, force=False):
        # save models to file
        if force or measure > self.best_measure:
            self.best_measure = measure
            if tune_params is not None:
                self.best_hyperparams = dict(tune_params)
                self.best_hyperparams['epochs'] = epoch + 1
                self.best_hyperparams['lr_scheduler'] = 'multi_step_lr'
                print("best hyperparams", self.best_hyperparams)
            print(f'Saving model: {measure:.2f}%')
            torch.save(self.model.state_dict(), os.path.join(self.config.models_path, 'tuned_model.pt'))

    def setup_optimizer(self, optimizer='sgd', lr=0.01, lr_scheduler='multi_step_lr', lr_gamma=0.1, lr_milestones=[10],
                        momentum=0.9, wd=1e-4, epochs=50, patience=10, mode='max'):
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=float(lr), momentum=float(momentum),
                                            weight_decay=wd)
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr), weight_decay=float(wd))

        print(lr_scheduler, lr_milestones)
        if lr_scheduler == 'multi_step_lr':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, lr_milestones, gamma=float(lr_gamma))
        elif lr_scheduler == 'reduce_lr_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=mode, patience=patience,
                                                                        verbose=True)

    def load_model(self, config, load=True, replace_classifier=True):
        model = select_model(config)
        if load:
            model.load_state_dict(torch.load(os.path.join(config.models_path, "model.pt")))
        if replace_classifier:
            modules = list(model.classifier.children())
            modules[-1] = nn.Linear(modules[-1].in_features, config['data']['test']['num_classes'])
            model.classifier = nn.Sequential(*modules)
        return model.to(config['device'])

    def load_best_tuned_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.config.models_path, "tuned_model.pt")))

    def tune(self, tune_params, validate=True):
        print(tune_params)
        if tune_params['lr_scheduler'] == 'reduce_lr_on_plateau':
            tune_params['lr_milestones'] = []
            last_lr = tune_params['lr']
        # train the model with labels on the validation data
        self.model.train()
        for epoch in range(tune_params['epochs']):
            pbar = tqdm(self.data.test_train_loader, desc=f'Epoch {epoch}')
            for data, targets in pbar:
                self.model.train()
                data, targets = data.to(self.config['device']), targets.to(self.config['device'])

                self.optimizer.zero_grad()
                train_y = self.model(data)
                loss = self.model.loss(train_y, targets)
                acc = 100. * count_acc(train_y, targets)
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(loss=loss.item(), acc=acc)

            if validate:
                val_loss, val_acc, val_acc_top5 = self.test_classifier(epoch, self.data.val_test_loader)
                print(f'Classifier val accuracy {val_acc:.2f}%')
                self.save_model(val_acc, epoch, tune_params)
            if tune_params['lr_scheduler'] == 'multi_step_lr':
                self.scheduler.step(epoch)
            elif tune_params['lr_scheduler'] == 'reduce_lr_on_plateau':
                if validate:
                    if tune_params['mode'] == 'max':
                        self.scheduler.step(val_acc)
                    elif tune_params['mode'] == 'min':
                        self.scheduler.step(val_loss)
                if self.scheduler.optimizer.param_groups[0]['lr'] != last_lr:
                    tune_params['lr_milestones'].append(epoch)
                    last_lr = self.scheduler.optimizer.param_groups[0]['lr']

            print(f"Current learning rate: {self.scheduler.optimizer.param_groups[0]['lr']}")

    def run_episodes(self, validate=True):
        for tune_params in self.episodes:
            epochs = tune_params['epochs']
            self.model = self.load_model(self.config)
            self.setup_optimizer(**tune_params)
            self.tune(tune_params, validate=validate)

    def retune(self, save_model=True):
        tune_params = self.best_hyperparams

        full_val_set = torch.utils.data.ConcatDataset([self.data.val_train_set, self.data.val_test_set])
        self.data.test_train_loader = torch.utils.data.DataLoader(full_val_set,
                                                           batch_size=self.config['finetuning']['batch_size'],
                                                           shuffle=True, drop_last=True)
        self.model = self.load_model(self.config)
        self.setup_optimizer(**tune_params)
        self.tune(tune_params, validate=False)
        if save_model:
            self.save_model(-1, -1, force=True)

    def test_classifier(self, epoch, data_loader):
        self.model.eval()
        test_loss = 0
        test_acc, test_acc_top5 = 0, 0
        num_batches = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(data_loader):
                num_batches += 1
                data, target = data.to(self.config['device']), target.to(self.config['device'])
                output = self.model(data)
                test_loss += self.model.loss(output, target).item()
                if 'imagenet' in self.config['data']['test_set']:
                    ta, ta_top5 = accuracy(output, target, topk=(1, 5))
                    test_acc += ta
                    test_acc_top5 += ta_top5
                else:
                    ta = 100. * count_acc(output, target)
                    test_acc += ta
        test_loss /= num_batches
        test_acc /= num_batches
        test_acc_top5 /= num_batches
        self.model.train()
        return test_loss, test_acc, test_acc_top5

def latest_dir(b='.'):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return max(result, key=os.path.getmtime)
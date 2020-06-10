import os

import numpy as np

import torch
from torch._utils import _accumulate
from torchvision import datasets, transforms
from torchvision.transforms import functional as FT


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Data():
    def __init__(self, config):
        self.config = config
        # set up data loaders
        self.make_data_loaders()

    def get_transform(self, name, split, resize, cropsize, channels, h_flips):
        normalizations = {
            'stl10': {'mean': [0.441, 0.427, 0.386], 'std': [0.25, 0.243, 0.251]},
            'VD-aircraft': {'mean': [0.486, 0.516, 0.541], 'std': [0.192, 0.188, 0.224]},
            'VD-cifar100': {'mean': [0.507, 0.487, 0.441], 'std': [0.261, 0.25, 0.27]},
            'VD-daimlerpedcls': {'mean': [0.482, 0.482, 0.482], 'std': [0.231, 0.231, 0.231]},
            'VD-dtd': {'mean': [0.529, 0.475, 0.429], 'std': [0.231, 0.219, 0.231]},
            'VD-gtsrb': {'mean': [0.339, 0.311, 0.32], 'std': [0.269, 0.259, 0.264]},
            'VD-imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            'VD-omniglot': {'mean': [0.081, 0.081, 0.081], 'std': [0.226, 0.226, 0.226]},
            'VD-svhn': {'mean': [0.438, 0.444, 0.473], 'std': [0.196, 0.199, 0.195]},
            'VD-ucf101': {'mean': [0.499, 0.498, 0.498], 'std': [0.101, 0.1, 0.099]},
            'VD-vgg-flowers': {'mean': [0.446, 0.391, 0.311], 'std': [0.271, 0.219, 0.247]}
        }
        train_set_name = self.config['data']['datasets'][0]
        if self.config['data']['training']['task'] == 'reconstruction':
            norm = {'mean': [0.5 for _ in range(channels)], 'std': [0.5 for _ in range(channels)]}
        elif train_set_name in normalizations:
            norm = normalizations[train_set_name]
        else:
            norm = {'mean': [0.5 for _ in range(channels)], 'std': [0.5 for _ in range(channels)]}
        print(f'Normalising {train_set_name}: {norm}')

        unnormalizer = UnNormalize(**norm)
        self.unnormalize = lambda images: torch.stack([unnormalizer(image) for image in images])

        if split == 'train':
            if 'no_crop' in self.config['data'] and self.config['data']['no_crop']:
                crop = transforms.CenterCrop(resize)
            elif cropsize == resize:
                crop = transforms.RandomCrop(cropsize, padding=4)
            else:
                crop = transforms.RandomCrop(cropsize)

            transforms_list = [
                transforms.Resize(resize),
                crop,
                transforms.ToTensor(),
                transforms.Normalize(**norm)
            ]
        else:
            transforms_list = [
                transforms.Resize(resize),
                transforms.CenterCrop(cropsize),
                transforms.ToTensor(),
                transforms.Normalize(**norm)
            ]
        if h_flips:
            transforms_list.insert(0, transforms.RandomHorizontalFlip())

        print(transforms_list, flush=True)
        transform = transforms.Compose(transforms_list)
        return transform, transforms_list

    def get_dataset(self, name, split):

        if self.config['data']['split_mode'] == 'none_train_val':
            if name == self.config['data']['test_set']:
                # since we will only use 'val' and 'test' we swap them for 'train' and 'val'
                if split == 'train':
                    raise Exception("You should not be using split='train' when using split_mode='none_train_val'")
                if split == 'val':
                    split = 'train'
                elif split == 'test':
                    split = 'val'

        transform, _ = self.get_transform(name, split, self.config['data']['resize'], self.config['data']['cropsize'],
            self.config['data']['channels'], self.config['data']['h_flips'])

        # MNISTs
        if name == 'mnist':
            d = DatasetWithValSplit(datasets.MNIST, os.path.join(self.config['data']['root'], 'MNIST'),
                transform=transform,
                split=split)()
        elif name == 'fashionmnist':
            d = DatasetWithValSplit(datasets.FashionMNIST, os.path.join(self.config['data']['root'], 'FashionMNIST'),
                transform=transform,
                split=split)()
        elif name == 'kmnist':
            d = DatasetWithValSplit(datasets.KMNIST, os.path.join(self.config['data']['root'], 'KMNIST'),
                transform=transform,
                split=split)()
        elif name == 'stl10':
            d = DatasetWithValSplit(datasets.STL10, os.path.join(self.config['data']['root'], 'STL10'),
                transform=transform,
                split=split)()

        # Visual Decathlon
        elif name == 'VD-aircraft':
            d = DatasetWithValSplit(VDDataset, os.path.join(self.config['data']['root'], 'VD', 'aircraft'),
                split=split,
                transform=transform)()
        elif name == 'VD-cifar100':
            d = DatasetWithValSplit(VDDataset, os.path.join(self.config['data']['root'], 'VD', 'cifar100'),
                split=split,
                transform=transform)()
        elif name == 'VD-daimlerpedcls':
            d = DatasetWithValSplit(VDDataset, os.path.join(self.config['data']['root'], 'VD', 'daimlerpedcls'),
                split=split,
                transform=transform)()
        elif name == 'VD-dtd':
            d = DatasetWithValSplit(VDDataset, os.path.join(self.config['data']['root'], 'VD', 'dtd'),
                split=split,
                transform=transform)()
        elif name == 'VD-gtsrb':
            d = DatasetWithValSplit(VDDataset, os.path.join(self.config['data']['root'], 'VD', 'gtsrb'),
                split=split,
                transform=transform)()
        elif name == 'VD-imagenet':
            d = DatasetWithValSplit(VDDataset, os.path.join(self.config['data']['root'], 'VD', 'imagenet'),
                split=split,
                transform=transform)()
        elif name == 'VD-omniglot':
            d = DatasetWithValSplit(VDDataset, os.path.join(self.config['data']['root'], 'VD', 'omniglot'),
                split=split,
                transform=transform)()
        elif name == 'VD-svhn':
            d = DatasetWithValSplit(VDDataset, os.path.join(self.config['data']['root'], 'VD', 'svhn'),
                split=split,
                transform=transform)()
        elif name == 'VD-ucf101':
            d = DatasetWithValSplit(VDDataset, os.path.join(self.config['data']['root'], 'VD', 'ucf101'),
                split=split,
                transform=transform)()
        elif name == 'VD-vgg-flowers':
            d = DatasetWithValSplit(VDDataset, os.path.join(self.config['data']['root'], 'VD', 'vgg-flowers'),
                split=split,
                transform=transform)()

        else:
            raise Exception(f'Incorrect dataset supplied: {name}')
        return d

    def construct_train_set(self):
        train_sets = []
        print("Constructing training set", flush=True)
        for i, name in enumerate(self.config['data']['datasets']):
            if self.config['data']['test_set'] == name:
                if len(self.config['data']['datasets']) > 1 or 'optimal' in self.config['exp']['name']:
                    # if the test domain is part of the training domain
                    # we can load the validation set as well.
                    train_domain = self.get_dataset(name, 'train')
                    val_train = self.get_dataset(name, 'val_train')
                    train_domain = torch.utils.data.ConcatDataset([train_domain, val_train])
                    print(f"- loaded val_train set separately from {name} (test domain)", flush=True)
                else:
                    # we only have one train domain and the same test domain
                    # so we only load train for the train set
                    train_domain = self.get_dataset(name, 'train')
                    train_domain = torch.utils.data.ConcatDataset([train_domain])
            else:
                # add any domain which is not the test domain
                # we add both train, val_train, val_test and test in these domains
                train_domain = self.get_dataset(name, 'train')
                test = self.get_dataset(name, 'test')
                try:
                    val_train = self.get_dataset(name, 'val_train')
                    val_test = self.get_dataset(name, 'val_test')
                    train_domain = torch.utils.data.ConcatDataset([train_domain, val_train, val_test, test])
                    print(f"- adding train, val_train, val_test and test from {name} (distracting domain)", flush=True)
                except:
                    train_domain = torch.utils.data.ConcatDataset([train_domain, test])
                    print(f"- adding train and test from {name} (distracting domain)", flush=True)
            train_sets.append(train_domain)

        # concatenate the datasets
        train_set = IndexConcatDataset(self.config, train_sets)

        ds_lens = [len(d) for d in train_sets]
        self.config['data']['dataset_len'] = ds_lens
        self.config.num_data_points = sum(ds_lens)
        print("Training set consists of:", flush=True)
        for l, d in zip(ds_lens, self.config['data']['datasets']):
            print(f'- {d}: {l}', flush=True)
        print(f'- Total: {self.config.num_data_points}', flush=True)

        return train_set

    def get_val_sets(self):
        print("Loading val_train and val_test sets", flush=True)
        val_train_set = self.get_dataset(self.config['data']['test_set'], 'val_train')
        #test_train_set = self.get_dataset(self.config['data']['test_set'], 'val_train')
        val_test_set = self.get_dataset(self.config['data']['test_set'], 'val_test')
        print(f"- loaded val_train and val_test sets separately from {self.config['data']['test_set']} (test domain)",
            flush=True)

        print("Val_train set consists of:", flush=True)
        print(f"- {self.config['data']['test_set']}: {len(val_train_set)}", flush=True)
        #print("Test train set consists of:", flush=True)
        #print(f"- {self.config['data']['test_set']}: {len(test_train_set)}", flush=True)
        print("Val_test set consists of:", flush=True)
        print(f"- {self.config['data']['test_set']}: {len(val_test_set)}", flush=True)

        return val_train_set, val_test_set

    def get_test_set(self):
        test_set = self.get_dataset(self.config['data']['test_set'], 'test')

        print(f"- loaded test set from {self.config['data']['test_set']} (test domain)", flush=True)
        print("Test set consists of:", flush=True)
        print(f"- {self.config['data']['test_set']}: {len(test_set)}", flush=True)

        return test_set

    def get_targets(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.config['data']['training']['batch_size'], shuffle=False)
        targets = []
        for _, target in data_loader:
            targets.append(target)
        targets = torch.cat(targets)
        return targets

    def make_data_loaders(self):
        # construct each dataset
        train_set = self.construct_train_set()
        val_train_set, val_test_set = self.get_val_sets()
        test_set = self.get_test_set()

        # create training data loader from the concatenated datasets
        train_loader = torch.utils.data.DataLoader(train_set,
            batch_size=self.config['data']['training']['batch_size'], drop_last=True, shuffle=True)

        # create validation loader
        if self.config['data']['validation']['task'] == 'prototypical-networks':
            targets = self.get_targets(val_train_set)
            val_train_loader = torch.utils.data.DataLoader(val_train_set,
                batch_sampler=CategoriesSampler(targets, 100, self.config['data']['validation']['num_classes_per_batch'],
                    (self.config['data']['validation']['num_data_per_class']
                        + self.config['data']['validation']['num_queries'])))
        else:
            val_train_loader = torch.utils.data.DataLoader(val_train_set,
                shuffle=True, batch_size=self.config['data']['validation']['batch_size'], drop_last=True)

        # create data loader for the set used to train logistic regression
        test_train_loader = torch.utils.data.DataLoader(val_train_set,
            shuffle=False, batch_size=self.config['data']['test']['batch_size'], drop_last=True)
        # create data loader for the validation set, used for model selection
        val_test_loader = torch.utils.data.DataLoader(val_test_set,
            shuffle=False, batch_size=self.config['data']['test']['batch_size'], drop_last=False)
        # create data loader for the real test set
        test_loader = torch.utils.data.DataLoader(test_set,
            shuffle=False, batch_size=self.config['data']['test']['batch_size'], drop_last=False)

        # keeps track of the data points still used for training
        self.idxs_kept = torch.arange(self.config.num_data_points)

        # make datasets accessible
        self.train_set = train_set
        self.val_train_set = val_train_set
        self.val_test_set = val_test_set
        self.test_set = test_set

        # make data loaders accessible
        self.train_loader = train_loader
        self.val_train_loader = val_train_loader
        self.val_test_loader = val_test_loader
        self.test_train_loader = test_train_loader
        self.test_loader = test_loader

    def prune(self, w, g, weights_model):
        mode = self.config['weights']['pruning']['mode']
        idxs = w.argsort()
        if mode == 'cdf':
            # if the density at epsilon is bigger than delta, then prune the data point
            cdf_values = weights_model.cdf(self.config['weights']['pruning']['epsilon'])
            idxs_to_keep = (cdf_values <= self.config['weights']['pruning']['delta']).nonzero().squeeze()
        elif mode == 'threshold':
            idxs_to_keep = (w > self.config['weights']['pruning']['epsilon']).nonzero()
        else:
            print('No pruning mode specified', flush=True)

        self.idxs_kept = np.intersect1d(self.idxs_kept, idxs_to_keep)

        proportion_left = (len(self.idxs_kept) / self.config.num_data_points)
        print(f"New size of dataset: {len(self.idxs_kept)} ({proportion_left * 100:.1f}%)", flush=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_loader.dataset,
            sampler=torch.utils.data.SubsetRandomSampler(self.idxs_kept),
            batch_size=self.config['data']['training']['batch_size'], drop_last=True)
        return proportion_left


class CategoriesSampler():
    """
    Sampler for the prototypical style loss
    """
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class IndexConcatDataset(torch.utils.data.ConcatDataset):
    """
    Returns the indices of the data points, so we can associate them with a weight
    """
    def __init__(self, config, datasets):
        super().__init__(datasets)
        self.config = config

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class DatasetWithValSplit():
    def __init__(self, dataset, root, split, transform):
        try:
            indices = [int(line.rstrip('\n')) for line in open(os.path.join(root, f'{split}.txt'), 'r')]
            print(f'Loading generated split {split}.txt')
            try:
                d = dataset(root, train=True, transform=transform)
            except:
                d = dataset(root, split='train', transform=transform)
            self.dataset = torch.utils.data.Subset(d, indices)
        except:
            if (dataset in [datasets.STL10] and split == 'train'):
                print('Loading unlabeled data')
                self.dataset = dataset(root, split='unlabeled', transform=transform)
            elif split == 'test':
                # since we don't have the test labels
                if '/VD/' in root:
                    split = 'val'
                print('Loading predefined test split')
                try:
                    self.dataset = dataset(root, train=False, transform=transform)
                except:
                    self.dataset = dataset(root, split=split, transform=transform)

    def __call__(self):
        return self.dataset


class VDDataset(datasets.ImageFolder):
    def __init__(self, root, split, transform):
        # since we don't have test labels there are no subfolders in 'test'
        if not split == 'test':
            root = os.path.join(root, split)
            valid_file_func = lambda x: x
        else:
            valid_file_func = lambda x: f'/{split}/' in x
        super().__init__(root, transform=transform, is_valid_file=valid_file_func)


class TrainTestSwapVDDataset(VDDataset):
    def __init__(self, root, split, transform):
        if split == 'train':
            split = 'test'
        elif split == 'test':
            split = 'train'
        super().__init__(root, split=split, transform=transform)


class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset, split):

        self.config  = config
        self.dataset = dataset

        self.train = (split == 'train')
        self.mode = {'train': 'training', 'val': 'validation', 'test': 'test'}[split]

    def __getitem__(self, index):
        if self.train:
            data, target, index = self.dataset.__getitem__(index)
        else:
            data, target = self.dataset.__getitem__(index)
        c = self.config['data'][self.mode]['num_classes']
        target = torch.randint(0, c, size=(1,)).item()
        data = FT.to_tensor(FT.to_pil_image(data).rotate(target * (360 / c)))
        if self.train:
            return data, target, index
        else:
            return data, target

    def __len__(self):
        return len(self.dataset)

def rotate(batch, c=4):
    new_data = torch.zeros((batch.shape[0] * c), *batch.shape[1:])
    new_targets = torch.zeros((batch.shape[0] * c), dtype=torch.long)
    for i, data in enumerate(batch):
        rotate_data = []
        rotate_targets = []
        for target in torch.arange(c):
            data = FT.to_tensor(FT.to_pil_image(data).rotate(target * (360 / c)))
            rotate_data.append(data)
        rotate_data = torch.stack(rotate_data)
        new_targets[i * c: (i + 1) * c] = torch.arange(c)
        new_data[i * c: (i + 1) * c] = rotate_data
    return new_data, new_targets

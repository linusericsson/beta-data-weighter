import os, shutil

import torch
from torchvision import datasets, transforms
from torch._utils import _accumulate
from torch import randperm

from data import *


def class_balance(dataset, num_classes, normalise=False):
    # check the balance of classes in the given dataset
    counts = torch.zeros(num_classes)
    for _, target in torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False):
        counts[target] += 1
    if normalise:
        counts /= len(dataset)
    return counts

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

dataset = {
    'mnist':        (datasets.MNIST, DatasetWithValSplit, '../data/MNIST', 'train', 10,
        ('train', 'val_train', 'val_test'), True, 500, 0),
    'fashionmnist': (datasets.FashionMNIST, DatasetWithValSplit, '../data/FashionMNIST', 'train', 10,
        ('train', 'val_train', 'val_test'), True, 500, 0),
    'kmnist':       (datasets.KMNIST, DatasetWithValSplit, '../data/KMNIST', 'train', 10,
        ('train', 'val_train', 'val_test'), True, 500, 0)
}

def get_targets(dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    targets = []
    for _, target in data_loader:
        targets.append(target)
    targets = torch.cat(targets)
    return targets

def stratified_split(dataset, split_names, num_classes, num_per_class, rest_split_num):

    targets = get_targets(dataset)
    splits = []

    class_indices = []
    class_perms = []

    names = list(split_names)
    last_split = names.pop(rest_split_num)
    names.append(last_split)

    # setup permutations for each class
    for i in range(num_classes):
        # get indices of elements of class i
        c = (targets == i).nonzero()
        class_indices.append(c)
        # random permutation of indices with class i
        perm = torch.randperm(len(c))
        class_perms.append(perm)

    # split into sets based on permutations
    for i in range(len(names)):
        split_indices = []
        indices_per_class = []
        for j in range(num_classes):
            if i == len(split_names) - 1:
                class_indices_for_split = class_indices[j][class_perms[j][i * num_per_class:]]
            else:
                class_indices_for_split = class_indices[j][class_perms[j][i* num_per_class:(i + 1) * num_per_class]]
            split_indices.extend(class_indices_for_split)
        if len(split_indices) == 0:
            splits.append(torch.tensor([]))
        else:
            splits.append(torch.cat(split_indices))

    last_split = splits.pop(-1)
    splits.insert(rest_split_num, last_split)

    return splits


for name, (pre_ds, post_ds, root, split, num_classes, split_names,
          stratified, num_per_class, rest_split_num) in dataset.items():
    #torch.manual_seed(0)
    print(name)

    # load training set
    try:
        split_set = pre_ds(root, split, transform=transform, download=True)
    except:
        split_set = pre_ds(root, split, transform=transform)

    split_indices = stratified_split(split_set, split_names, num_classes, num_per_class, rest_split_num)

    # save splits into files within the dataset root folder
    for i, (split, set_name) in enumerate(zip(split_indices, split_names)):
        with open(os.path.join(root, f'{set_name}.txt'), 'w') as file:
            for index in split:
                file.write(f'{index}\n')

    # load train, val and test using the parent class which deals with the train/val splits
    train_set = post_ds(pre_ds, root, 'train', transform=transform)()
    val_train_set = post_ds(pre_ds, root, 'val_train', transform=transform)()
    val_test_set = post_ds(pre_ds, root, 'val_test', transform=transform)()
    test_set = post_ds(pre_ds, root, 'test', transform=transform)()

    print("\t train", len(train_set))
    #print("\t\t", class_balance(train_set, num_classes))
    print("\t val_train", len(val_train_set))
    print("\t\t", class_balance(val_train_set, num_classes))
    print("\t val_test", len(val_test_set))
    print("\t\t", class_balance(val_test_set, num_classes))
    print("\t test", len(test_set))
    print("\t\t", class_balance(test_set, num_classes))
    
    print(len(get_targets(val_train_set)))

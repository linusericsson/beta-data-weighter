import torch


def count_acc(logits, label):
    pred = logits.argmax(dim=1)
    return pred.eq(label.view_as(pred)).to(torch.float32).mean().item()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_printer(msg):
    """This function returns a printer function, that prints information about a  tensor's
    gradient. Used by register_hook in the backward pass.
    """
    def printer(tensor):
        if tensor.nelement() == 1:
            print(f"{tensor.item()}: {msg}", flush=True)
        else:
            print(f"{tensor.mean().item()}: {msg}", flush=True)
    return printer


def register_hook(tensor, msg):
    """Utility function to call retain_grad and Pytorch's register_hook
    in a single line
    """
    tensor.retain_grad()
    tensor.register_hook(get_printer(msg))


def print_params(model, meta_model, weight_model, msg):
    print(msg, flush=True)
    print('', meta_model.params[list(meta_model.params.keys())[0]].mean().item(), 'meta_model params weights', flush=True)
    print('', list(model.parameters())[0].mean().item(), 'model weights', flush=True)
    print('', list(weight_model.parameters())[0].mean().item(), 'weights weights', flush=True)


def gpu_info(config):
    if 'cuda' in config['device']:
        t = torch.cuda.get_device_properties(config['device']).total_memory
        c = torch.cuda.memory_cached(config['device'])
        a = torch.cuda.memory_allocated(config['device'])
        f = c - a  # free inside cache
        free = t - c
        print(f'GPU memory', flush=True)
        print(f'\t total: {t / 1e9:.2f} GB', flush=True)
        print(f'\t cached: {c / 1e9:.2f} GB', flush=True)
        print(f'\t allocated: {a / 1e9:.2f} GB', flush=True)
        print(f'\t free in cache: {f / 1e9:.2f} GB', flush=True)
        print(f'\t free in cache: {f / 1e9:.2f} GB', flush=True)
        print(f'\t free: {free / 1e9:.2f} GB', flush=True)
    else:
        print('Using CPU')

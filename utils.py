import os
import numpy as np

join_path = os.path.join


def center_print(*args, around='*', repeat_around=10):
    num = repeat_around
    s = around
    print(num * s, end=' ')
    print(*args, end=' ')
    print(num * s)


def tensor2image(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.squeeze(0).permute([1, 2, 0]).numpy()
    return (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))


def tensor2text(text_tensor, vocab):
    pad_idx = vocab.stoi['<pad>']
    if text_tensor.is_cuda:
        text_tensor = text_tensor.cpu()
    text_tensor = text_tensor.numpy()
    text_string = ' '.join([vocab.itos[idx] if idx != pad_idx else '' for idx in text_tensor]) + '.'
    text_string = text_string[0].upper() + text_string[1:]
    return text_string


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Records(object):
    def __init__(self):
        self.records = dict()

    def record(self, r, n=1):
        for k, v in r.items():
            if k not in self.records:
                self.records[k] = AverageMeter()
            self.records[k].update(v, n=n)

    def __getitem__(self, k):
        return self.records[k].avg

    def __str__(self):
        return ' | '.join([str(k) + ': ' + str(round(v.avg, 4)) for k, v in self.records.items()])

    def reset(self):
        self.records.clear()

    def pop(self, k):
        self.records.pop(k)

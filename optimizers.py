from torch.optim import SGD

from utils import center_print

_name2optimizer = {
    'sgd': SGD
}


def get_optimizer(name, *args, **kwargs):
    if name not in _name2optimizer:
        raise ValueError('Optimizer %s not define' % name)
    center_print('Using optimizer:', name)
    center_print('with *args:', *args, repeat_around=3)
    center_print('with **kwargs:', str(kwargs), repeat_around=3)
    optimizer = _name2optimizer[name]
    return optimizer(*args, **kwargs)

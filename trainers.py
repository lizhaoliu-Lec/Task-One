import time
import os
from tensorboardX import SummaryWriter

import torch

from optimizers import get_optimizer
from utils import center_print, join_path, Records


def get_run_id(run_id=None):
    time_format = '%Y-%m-%d...%H.%M.%S'
    default_run_id = time.strftime(time_format, time.localtime(time.time()))
    run_id = default_run_id or run_id
    return run_id


class BaseTrainer(object):
    def __init__(self,
                 model,
                 name,
                 loss_func,
                 optimizer,
                 train_loader,
                 train_steps=1000,
                 val_every=100,
                 val_loader=None,
                 log_every=10,
                 run_base_dir='runs',
                 run_id=None,
                 cuda=None,
                 tensorboard=True):
        self.model = model
        self.name = name
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_steps = train_steps
        self.val_every = val_every
        self.log_every = log_every

        self.global_step = 0
        self.run_base_dir = run_base_dir
        if not os.path.exists(self.run_base_dir):
            os.makedirs(self.run_base_dir)

        self.run_id = get_run_id(run_id)

        self.run_dir = join_path(self.run_base_dir, self.name, self.run_id)
        center_print('Saving records on: ', self.run_dir)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.cuda = cuda
        if self.cuda is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cuda)
            if not torch.cuda.is_available():
                center_print('Waring: trying to use cuda device %d, but cuda is not available' % cuda)
                center_print('Using CPU instead.')
                self.cuda = None
            center_print('Using cuda device: %d' % cuda)

        if self.cuda is not None:
            self.model = model.cuda()

        self.optimizer = get_optimizer(optimizer.pop('name'),
                                       self.model.parameters(),
                                       **optimizer)

        self.records = Records()

        self.tensorboard = None
        if tensorboard:
            self.tensorboard = SummaryWriter(log_dir=self.run_dir)

    def save_checkpoint(self, is_best=False):
        model_name = 'checkpoint_latest' if is_best else 'checkpoint_best'
        save_path = join_path(self.run_dir, model_name + '.bin')
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }, save_path)

    def load_checkpoint(self, get_best=True):
        model_name = 'checkpoint_latest' if not get_best else 'checkpoint_best'
        load_path = join_path(self.run_dir, model_name + '.bin')
        load_dict = torch.load(load_path)
        self.model.load_state_dict(load_dict['model'])
        self.optimizer.load_state_dict((load_dict['optimizer']))
        self.global_step = load_dict['global_step']

    def train(self):
        center_print('Training Process Begins')
        for step_idx in range(1, self.train_steps + 1):
            self.global_step = step_idx
            if self.global_step == self.train_steps:
                center_print('Training Process Ends.')
                self.tensorboard.close()
                return
            self.step()

            if self.global_step % self.val_every == 0 and self.val_loader is not None:
                self.validation()

    def step(self):
        raise NotImplementedError

    def validation(self):
        raise NotImplementedError

    @staticmethod
    def to_cuda(*args):
        length = len(args)
        args = [a.cuda() for a in args]
        if length == 1:
            return args[0]
        else:
            return args


class BaseTester(object):
    ...


if __name__ == '__main__':
    print(dir(BaseTester))
    print(BaseTester.__class__)
    print(BaseTester.__repr__)
    print(BaseTester.__str__)
    print(BaseTester.__doc__)
    print(BaseTester.__format__)

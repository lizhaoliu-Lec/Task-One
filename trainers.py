import time
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

import torch

from utils import join_path, Records, center_print, tensor2image
from optimizers import get_optimizer


def get_run_id(run_id=None):
    if run_id is None:
        time_format = '%Y-%m-%d...%H.%M.%S'
        default_run_id = time.strftime(time_format, time.localtime(time.time()))
        return default_run_id
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
                 tensorboard=True,
                 debug=False):
        self.model = model
        self.name = name
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_steps = train_steps
        self.val_every = val_every
        self.log_every = log_every
        self.debug = debug
        if self.debug:
            center_print('This debug mode, no tensorboard will be record')

        self.global_step = 0
        self.run_base_dir = run_base_dir
        if not os.path.exists(self.run_base_dir):
            os.makedirs(self.run_base_dir)

        self.run_id = get_run_id(run_id)

        self.run_dir = None
        if not self.debug:
            self.run_dir = join_path(self.run_base_dir, self.name, self.run_id)
            center_print('Saving records on: ', self.run_dir)
            if not os.path.exists(self.run_dir):
                os.makedirs(self.run_dir)
            else:
                raise ValueError('duplicated run id `%s`' % self.run_id)

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
        if tensorboard and not debug:
            self.tensorboard = SummaryWriter(log_dir=self.run_dir)

    def save_checkpoint(self, is_best=False):
        if not self.debug:
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
                if self.tensorboard is not None:
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


class ImageClassifierTrainer(BaseTrainer):

    def __init__(self,
                 dataset_name='CIFAR10',
                 num_plot=4,
                 plot_every=100,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.num_plot = num_plot
        self.plot_every = plot_every
        self.best_accuracy = 0
        self.plot_base_dir = dataset_name

    @staticmethod
    def accuracy(labels, predicts):
        total = labels.size(0)
        _, predicts = torch.max(predicts.data, 1)
        correct = (predicts == labels).sum().item()
        return correct / total

    def step(self):
        self.model.train()
        inputs, labels = next(iter(self.train_loader))
        if self.cuda is not None:
            inputs, labels = self.to_cuda(inputs, labels)

        outputs = self.model(inputs)
        loss = self.loss_func(outputs, labels)
        loss.backward()
        self.optimizer.step()

        acc = self.accuracy(labels, outputs)
        loss_name = 'train/loss'
        accuracy_name = 'train/accuracy'
        r = {loss_name: loss.item(), accuracy_name: acc}
        self.records.record(r, n=inputs.size()[0])

        if self.global_step % self.log_every == 0 or self.global_step == 1:
            print('Step: %d,' % self.global_step, self.records)
            self.add_scalar(d={loss_name: loss.item(), accuracy_name: acc})

        if self.global_step % self.plot_every == 0:
            self.add_images(images=inputs, labels=labels, predicts=outputs, train=True)

    def add_scalar(self, d):
        if self.tensorboard is not None:
            for k, v in d.items():
                self.tensorboard.add_scalar(tag=k, scalar_value=v, global_step=self.global_step)

    def add_images(self, images, labels, predicts, train=True):
        def _plot_image(tensorboard, image_tensor, tag, title):
            image = tensor2image(image_tensor)
            plotted_image = plt.figure()
            plt.title(title)
            plt.imshow(image)
            plt.axis('off')
            tensorboard.add_figure(tag=tag, figure=plotted_image, global_step=self.global_step)

        if self.tensorboard is not None:
            _, predicts = torch.max(predicts.data, 1)
            batch_size = images.size()[0]
            batch_indexes = [_ for _ in range(batch_size)]
            random.shuffle(batch_indexes)
            plot_indexes = batch_indexes[:self.num_plot]
            for plot_idx in plot_indexes:
                ground_truth = labels[plot_idx].item()
                predict = predicts[plot_idx].item()
                _plot_image(self.tensorboard, images[plot_idx], tag='/'.join(
                    [self.plot_base_dir, 'Class %d' % ground_truth, 'train' if train else 'validation']),
                            title='Prediction: %d' % predict)

    def validation(self):
        val_records = Records()
        center_print('Performing validation')
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.val_loader):
                inputs, labels = data
                if self.cuda is not None:
                    inputs, labels = self.to_cuda(inputs, labels)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                acc = self.accuracy(labels, outputs)
                loss_name = 'val/loss'
                accuracy_name = 'val/accuracy'
                r = {loss_name: loss.item(), accuracy_name: acc}
                val_records.record(r, n=inputs.size()[0])
        print('Validation Result: ', val_records)
        val_acc = val_records[accuracy_name]
        loss = val_records[loss_name]
        self.add_scalar(d={loss_name: loss, accuracy_name: val_acc})
        if self.global_step % self.plot_every == 0:
            self.add_images(images=inputs, labels=labels, predicts=outputs, train=False)

        if val_acc > self.best_accuracy:
            center_print('Better model occurs', around='!')
            self.best_accuracy = val_acc
            self.save_checkpoint(is_best=True)
        self.save_checkpoint(is_best=False)
        print()


class InstanceSegmentorTrainer(BaseTester):
    ...


if __name__ == '__main__':
    print(dir(BaseTester))
    print(BaseTester.__class__)
    print(BaseTester.__repr__)
    print(BaseTester.__str__)
    print(BaseTester.__doc__)
    print(BaseTester.__format__)

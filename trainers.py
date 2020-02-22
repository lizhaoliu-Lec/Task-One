import time
import os
import math
import sys
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from utils import join_path, Records, center_print, tensor2image, tensor2text
from references.detection.utils import MetricLogger, warmup_lr_scheduler, reduce_dict
from references.detection.coco_utils import get_coco_api_from_dataset
from references.detection.coco_eval import CocoEvaluator
from references.detection.engine import _get_iou_types


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
                 optimizer,
                 train_loader,
                 train_steps=1000,
                 val_every=100,
                 val_loader=None,
                 log_every=10,
                 lr_scheduler=None,
                 run_base_dir='runs',
                 run_id=None,
                 cuda=None,
                 tensorboard=True,
                 debug=False):
        self.model = model
        self.name = name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
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
            # reset the record after every epoch
            if self.global_step % len(self.train_loader) == 0:
                self.records.reset()

            self.step()

            if self.global_step % self.val_every == 0 and self.val_loader is not None:
                self.validation()

        center_print('Training Process Ends.')
        if self.tensorboard is not None:
            self.tensorboard.close()

    def step(self):
        raise NotImplementedError

    @torch.no_grad()
    def validation(self):
        raise NotImplementedError

    def to_device(self, *args):
        length = len(args)
        args = [a.cuda() if self.cuda is not None else a for a in args]
        if length == 1:
            return args[0]
        else:
            return args

    @property
    def epoch(self):
        return self.global_step / len(self.train_loader)


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
        inputs, labels = self.to_device(inputs, labels)

        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        self.optimizer.step()

        acc = self.accuracy(labels, outputs)
        loss_name = 'train/loss'
        accuracy_name = 'train/accuracy'
        r = {loss_name: loss.item(), accuracy_name: acc}
        self.records.record(r)

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

    @torch.no_grad()
    def validation(self):
        val_records = Records()
        center_print('Performing validation')
        self.model.eval()
        loss_name = 'val/loss'
        accuracy_name = 'val/accuracy'
        inputs, labels, outputs = None, None, None
        for data in tqdm(self.val_loader):
            inputs, labels = data
            inputs, labels = self.to_device(inputs, labels)
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
            acc = self.accuracy(labels, outputs)
            r = {loss_name: loss.item(), accuracy_name: acc}
            val_records.record(r)
        print('Validation Result: ', val_records)
        val_acc = val_records[accuracy_name]
        loss = val_records[loss_name]
        self.add_scalar(d={loss_name: loss, accuracy_name: val_acc})
        self.add_images(images=inputs, labels=labels, predicts=outputs, train=False)

        if val_acc > self.best_accuracy:
            center_print('Better model occurs', around='!')
            self.best_accuracy = val_acc
            self.save_checkpoint(is_best=True)
        self.save_checkpoint(is_best=False)
        print()


class InstanceSegmentorTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = MetricLogger(delimiter=" ")

    def step(self):
        self.model.train()
        header = 'Epoch: [{}]'.format(int(self.global_step / len(self.train_loader)))
        images, targets = next(
            self.records.log_every(self.train_loader, self.global_step % len(self.train_loader), self.log_every,
                                   header))
        images = self.to_device(*list(image for image in images))
        targets = [{k: self.to_device(v) for k, v in t.items()} for t in targets]

        lr_scheduler = None
        if self.epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self.train_loader) - 1)

            lr_scheduler = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

        loss_dict = self.model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=self.epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch=self.epoch)

        self.records.update(loss=losses_reduced, **loss_dict_reduced)
        self.records.update(lr=self.optimizer.param_groups[0]["lr"])

    @torch.no_grad()
    def validation(self):
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'

        coco = get_coco_api_from_dataset(self.val_loader.dataset)
        iou_types = _get_iou_types(self.model)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        i = 0
        for images, targets in metric_logger.log_every(self.val_loader, i, 100, header):
            i += 1
            images = self.to_device(*list(image for image in images))
            targets = [{k: self.to_device(v) for k, v in t.items()} for t in targets]

            torch.cuda.synchronize()
            model_time = time.time()
            outputs = self.model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        return coco_evaluator


class TextClassifierTrainer(BaseTrainer):
    def __init__(self,
                 dataset_name='AGNews',
                 num_write=4,
                 write_every=100,
                 vocab=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.num_write = num_write
        self.write_every = write_every
        self.vocab = vocab
        self.best_accuracy = 0
        self.write_base_dir = dataset_name

    def add_scalar(self, d):
        if self.tensorboard is not None:
            for k, v in d.items():
                self.tensorboard.add_scalar(tag=k, scalar_value=v, global_step=self.global_step)

    @staticmethod
    def accuracy(labels, predicts):
        total = labels.size(0)
        _, predicts = torch.max(predicts.data, 1)
        correct = (predicts == labels).sum().item()
        return correct / total

    def step(self):
        self.model.train()
        inputs, labels = next(iter(self.train_loader))
        inputs, labels = self.to_device(inputs, labels)
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        self.optimizer.step()

        acc = self.accuracy(labels, outputs)

        loss_name = 'train/loss'
        accuracy_name = 'train/accuracy'
        r = {loss_name: loss.item(), accuracy_name: acc}
        self.records.record(r)

        if self.global_step % self.log_every == 0 or self.global_step == 0:
            print('Step: %d,' % self.global_step, self.records)
            self.add_scalar(d={loss_name: loss.item(), accuracy_name: acc})

        if self.global_step % self.write_every == 0:
            self.add_texts(texts=inputs, labels=labels, predicts=outputs, train=True)

    def add_texts(self, texts, labels, predicts, train=True):
        def _write_text(tensorboard, vocab, text_tensor, tag, suffix=None):
            text_string = tensor2text(text_tensor, vocab)
            if suffix is not None:
                text_string = text_string + ' <===> ' + suffix
            tensorboard.add_text(tag=tag, text_string=text_string, global_step=self.global_step)

        if self.tensorboard is not None:
            _, predicts = torch.max(predicts.data, 1)
            batch_size = texts.size()[0]
            batch_indexes = [_ for _ in range(batch_size)]
            random.shuffle(batch_indexes)
            write_indexes = batch_indexes[:self.num_write]

            for write_idx in write_indexes:
                ground_truth = labels[write_idx].item()
                predict = predicts[write_idx].item()
                _write_text(self.tensorboard, self.vocab, texts[write_idx], tag='/'.join(
                    [self.write_base_dir, 'Class %d' % ground_truth, 'train' if train else 'validation']),
                            suffix='Predict: %d' % predict)

    @torch.no_grad()
    def validation(self):
        val_records = Records()
        center_print('Performing validation')
        self.model.eval()
        loss_name = 'val/loss'
        accuracy_name = 'val/accuracy'
        inputs, labels, outputs = None, None, None
        for data in tqdm(self.val_loader):
            inputs, labels = data
            inputs, labels = self.to_device(inputs, labels)
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
            acc = self.accuracy(labels, outputs)
            r = {loss_name: loss.item(), accuracy_name: acc}
            val_records.record(r)
        print('Validation Result: ', val_records)
        val_acc = val_records[accuracy_name]
        loss = val_records[loss_name]
        self.add_scalar(d={loss_name: loss, accuracy_name: val_acc})
        self.add_texts(texts=inputs, labels=labels, predicts=outputs, train=True)

        if val_acc > self.best_accuracy:
            center_print('Better model occurs', around='!')
            self.best_accuracy = val_acc
            self.save_checkpoint(is_best=True)
        self.save_checkpoint(is_best=False)
        print()


if __name__ == '__main__':
    print(dir(BaseTester))
    print(BaseTester.__class__)
    print(BaseTester.__repr__)
    print(BaseTester.__str__)
    print(BaseTester.__doc__)
    print(BaseTester.__format__)

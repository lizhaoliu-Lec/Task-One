from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from trainers import BaseTrainer
from utils import join_path, Records, center_print, tensor2image
from models import SmallCNN


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

    def plot_image(self, image_tensor, tag, title):
        image = tensor2image(image_tensor)
        plot_image = plt.figure()
        plt.title(title)
        plt.imshow(image)
        plt.axis('off')
        self.tensorboard.add_figure(tag=tag, figure=plot_image, global_step=self.global_step)

    @staticmethod
    def accuracy(labels, predicts):
        total = labels.size(0)
        _, predicts = torch.max(predicts.data, 1)
        correct = (predicts == labels).sum().item()
        return 100 * correct / total

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
        if self.tensorboard is not None:
            _, predicts = torch.max(predicts.data, 1)
            batch_size = images.size()[0]
            batch_indexes = [_ for _ in range(batch_size)]
            random.shuffle(batch_indexes)
            plot_indexes = batch_indexes[:self.num_plot]
            for plot_idx in plot_indexes:
                ground_truth = labels[plot_idx].item()
                predict = predicts[plot_idx].item()
                self.plot_image(images[plot_idx], tag='/'.join(
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
                val_records.record(r)
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


def main():
    #########################
    # (0) hard code configs #
    #########################
    DATA_BASE_DIR = join_path('datasets', 'cifar10')
    BATCH_SIZE = 512
    NUM_WORKERS = 8
    TRAIN_STEPS = 10000
    VAL_EVERY = 100
    LOG_EVERY = 50
    NAME = 'SimpleCNN'
    CUDA = 0
    RUN_ID = 'example'
    PLOT_EVERY = 500

    #######################
    # (1) Define datasets #
    #######################
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root=DATA_BASE_DIR, train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=NUM_WORKERS)

    test_set = torchvision.datasets.CIFAR10(root=DATA_BASE_DIR, train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=NUM_WORKERS)

    ####################
    # (2) Define model #
    ####################
    model = SmallCNN()

    ############################
    # (3) Define loss function #
    ############################
    loss_func = nn.CrossEntropyLoss()

    ########################
    # (4) Define optimizer #
    ########################
    optimizer = {'name': 'sgd',
                 'lr': 0.001,
                 'momentum': 0.9}

    ####################
    # (5) Init trainer #
    ####################
    trainer = ImageClassifierTrainer(model=model,
                                     name=NAME,
                                     loss_func=loss_func,
                                     optimizer=optimizer,
                                     train_loader=train_loader,
                                     train_steps=TRAIN_STEPS,
                                     val_every=VAL_EVERY,
                                     val_loader=test_loader,  # use test set for validation
                                     log_every=LOG_EVERY,
                                     run_id=RUN_ID,
                                     cuda=CUDA,
                                     plot_every=PLOT_EVERY)

    ######################
    # (6) Begin Training #
    ######################
    trainer.train()


if __name__ == '__main__':
    main()

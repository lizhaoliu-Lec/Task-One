import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from trainers import ImageClassifierTrainer
from utils import join_path
from models import SmallInstanceSegmentor
from engine import train_one_epoch, evaluate



def main():
    #########################
    # (0) hard code configs #
    #########################
    DATA_BASE_DIR = join_path('datasets', 'PennFudan')
    DATA_SET_NAME = 'PennFudan'
    BATCH_SIZE = 2
    NUM_WORKERS = 4
    TRAIN_STEPS = 1000
    VAL_EVERY = 100
    LOG_EVERY = 50
    NAME = 'SmallClassifier'
    CUDA = 0
    RUN_ID = 'example'
    PLOT_EVERY = 500
    NUM_CLASSES = 10
    DEBUG = True

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
    model = SmallClassifier(num_classes=NUM_CLASSES)

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
                                     dataset_name=DATA_SET_NAME,
                                     train_loader=train_loader,
                                     train_steps=TRAIN_STEPS,
                                     val_every=VAL_EVERY,
                                     val_loader=test_loader,  # use test set for validation
                                     log_every=LOG_EVERY,
                                     run_id=RUN_ID,
                                     cuda=CUDA,
                                     plot_every=PLOT_EVERY,
                                     debug=DEBUG)

    ######################
    # (6) Begin Training #
    ######################
    trainer.train()


if __name__ == '__main__':
    main()
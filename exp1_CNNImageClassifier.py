import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from trainers import ImageClassifierTrainer
from utils import join_path
from models import SmallClassifier
from datasets import CIFAR10


def main():
    #########################
    # (0) hard code configs #
    #########################
    DATA_BASE_DIR = join_path('datasets', 'cifar10')
    DATA_SET_NAME = 'CIFAR10'
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    TRAIN_STEPS = 1000
    VAL_EVERY = 100
    LOG_EVERY = 50
    NAME = 'SmallCNN'
    CUDA = 3
    RUN_ID = 'example'
    PLOT_EVERY = 500
    NUM_CLASSES = 10
    DEBUG = False

    #######################
    # (1) Define datasets #
    #######################
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
             mean=(0.4914, 0.4822, 0.4465),
             std=(0.2470, 0.2435, 0.2616)
         )])

    train_set = CIFAR10(root=DATA_BASE_DIR, train=True,
                        download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)

    test_set = CIFAR10(root=DATA_BASE_DIR, train=False,
                       download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    ####################
    # (2) Define model #
    ####################
    model = SmallClassifier(num_classes=NUM_CLASSES)

    ###############################################################
    # (3) Define loss function inside the Trainer's step function #
    ###############################################################

    ########################
    # (4) Define optimizer #
    ########################
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

    ####################
    # (5) Init trainer #
    ####################
    trainer = ImageClassifierTrainer(model=model,
                                     name=NAME,
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
